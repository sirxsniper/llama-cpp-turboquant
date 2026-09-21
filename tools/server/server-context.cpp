#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "../../src/llama-ext.h" // [TAG_SPEC_PREFILL_TAIL_EXTRACT] llama_set_layer_inp_extract

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <cinttypes>
#include <deque>
#include <exception>
#include <memory>
#include <filesystem>
#include <mutex>
#include <random>
#include <utility>
#include <thread>
#include <chrono>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_gen_rows = 0; // [TAG_POOL_PREEMPT] leading sampled + draft rows of generating slots
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        n_gen_rows = 0;
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

//
// [TAG_MTMD_ASYNC_ENCODE] media encoding off the server loop
//
// Gate 8 (production --no-mmproj-offload): a 4000-token image took 346 s to encode on the CPU, and three text
// streams on the other slots stalled for 342 s of it. The encode ran inside yield_to_queue(), which only lets
// the queue answer /metrics and /slots in the meantime: every other task is declined until the yield ends, and
// no slot can decode, because the thread that builds and decodes the batch is the one running the encode.
//
// Here one encoder thread owns every mtmd encode call. A clip context is not thread-safe, so it runs one job at
// a time, first in first out. A slot whose prompt reaches a media chunk that is not encoded yet stays in
// SLOT_STATE_PROCESSING_PROMPT and adds no rows to the batch, so the loop keeps serving the other slots. When
// the job is done, the main thread decodes the embeddings into the llama context through the same
// process_mtmd_chunk_decode() as the synchronous path (llama_decode never leaves the main thread), so the
// slot's own token and position sequence, its checkpoints and the drafter repair are unchanged.
//
// A job owns deep copies of its chunks and never touches the slot or the task, so a request that is cancelled
// while its media encodes cannot leave the encoder with a dangling pointer. A cancelled job is skipped if it
// has not started, otherwise it runs to the end (a clip encode cannot be interrupted) and its result is dropped.
//
// Used when the projector runs on the CPU while the text model runs on a GPU, or on a GPU the text model does not
// use (an AMD iGPU through Vulkan, -mmdev igpu). On a CUDA device the encode is fast and a second thread driving CUDA is avoided, so it stays
// synchronous. MTMD_ASYNC_ENCODE=0 keeps the synchronous encode everywhere, MTMD_ASYNC_ENCODE=1 also allows a
// non-CUDA device the text model uses. Only llama-server is affected, the mtmd library and tools are not.
//
struct server_media_job {
    enum job_state : int {
        JOB_QUEUED    = 0,
        JOB_RUNNING   = 1,
        JOB_DONE      = 2, // the embeddings are ready
        JOB_FAILED    = 3,
        JOB_CANCELLED = 4, // dropped before it started
    };

    int id_slot = -1;
    int id_task = -1;

    // start index of each chunk in the task's tokens, and the job's own copy of that chunk (ascending)
    std::vector<size_t>                idxs;
    std::vector<mtmd::input_chunk_ptr> chunks;
    mtmd::batch_ptr                    mbatch;       // its entries point into `chunks`
    size_t                             n_tokens = 0; // embedding rows over all chunks

    std::atomic<int>     state       { JOB_QUEUED };
    std::atomic<bool>    cancelled   { false };
    std::atomic<int64_t> t_submit_us { 0 };
    std::atomic<int64_t> t_start_us  { 0 };
    std::atomic<int64_t> t_end_us    { 0 };

    bool finished() const {
        return state.load(std::memory_order_acquire) >= JOB_DONE;
    }

    bool holds(size_t idx) const {
        return std::find(idxs.begin(), idxs.end(), idx) != idxs.end();
    }

    // the embeddings of the chunk that starts at `idx`; nullptr unless the job is done and holds that chunk
    float * get_embd(size_t idx) {
        if (state.load(std::memory_order_acquire) != JOB_DONE) {
            return nullptr;
        }
        for (size_t i = 0; i < idxs.size(); ++i) {
            if (idxs[i] == idx) {
                return mtmd_batch_get_output_embd(mbatch.get(), chunks[i].get());
            }
        }
        return nullptr;
    }

    const char * state_str() const {
        switch (state.load(std::memory_order_acquire)) {
            case JOB_QUEUED:    return "queued";
            case JOB_RUNNING:   return "encoding";
            case JOB_DONE:      return "done";
            case JOB_FAILED:    return "failed";
            case JOB_CANCELLED: return "cancelled";
        }
        return "unknown";
    }
};

// [TAG_MTMD_ASYNC_ENCODE] the encoder thread, see server_media_job
struct server_media_encoder {
    ~server_media_encoder() {
        stop();
    }

    bool is_running() const {
        return thread.joinable();
    }

    // throws std::system_error if the thread cannot be created
    void start() {
        if (thread.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            req_stop = false;
            busy     = false;
        }
        thread = std::thread([this]() { loop(); });
    }

    // drops the jobs that have not started and waits for the running one, which cannot be interrupted
    void stop() {
        if (!thread.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            req_stop = true;
            for (auto & job : queue) {
                job->state.store(server_media_job::JOB_CANCELLED, std::memory_order_release);
            }
            queue.clear();
        }
        cv_work.notify_all();
        cv_done.notify_all();
        thread.join();
    }

    void submit(std::shared_ptr<server_media_job> job) {
        job->t_submit_us.store(ggml_time_us());
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(std::move(job));
        }
        cv_work.notify_one();
    }

    // returns as soon as one of the jobs is finished, or after the timeout
    void wait_any(const std::vector<const server_media_job *> & jobs, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait_for(lock, timeout, [&]() {
            if (req_stop) {
                return true;
            }
            for (const auto * job : jobs) {
                if (job->finished()) {
                    return true;
                }
            }
            return false;
        });
    }

    // returns as soon as no job is queued or running, or after the timeout
    void wait_idle(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait_for(lock, timeout, [&]() {
            return req_stop || (queue.empty() && !busy);
        });
    }

    // jobs queued or running, including a cancelled job that is still running
    int n_jobs() {
        std::lock_guard<std::mutex> lock(mutex);
        return (int) queue.size() + (busy ? 1 : 0);
    }

private:
    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv_work; // the encoder thread waits here for a job
    std::condition_variable cv_done; // the server loop waits here for a finished job
    std::deque<std::shared_ptr<server_media_job>> queue;
    bool req_stop = false;
    bool busy     = false;

    void loop() {
        while (true) {
            std::shared_ptr<server_media_job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_work.wait(lock, [&]() { return req_stop || !queue.empty(); });
                if (req_stop) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
                if (job->cancelled.load()) {
                    job->state.store(server_media_job::JOB_CANCELLED, std::memory_order_release);
                    // wait_idle() may be waiting for the queue to drain: tell it now, not at its timeout
                    lock.unlock();
                    cv_done.notify_all();
                    continue;
                }
                busy = true;
                job->state.store(server_media_job::JOB_RUNNING, std::memory_order_release);
            }

            job->t_start_us.store(ggml_time_us());

            // [TAG_MTMD_ENCODE_CATCH] mtmd_batch_encode already turns a std::exception from the encoder (a Vulkan
            // DeviceLost on the iGPU is a vk::SystemError) into an error code. Nothing else may escape this thread
            // either, that would terminate the process: it fails only this job, and so only its request.
            int32_t res = 1;
            try {
                res = mtmd_batch_encode(job->mbatch.get());
            } catch (const std::exception & e) {
                SRV_ERR("[TAG_MTMD_ASYNC_ENCODE] exception while encoding media for slot %d, task %d: %s\n",
                        job->id_slot, job->id_task, e.what());
                res = 1;
            } catch (...) {
                SRV_ERR("[TAG_MTMD_ASYNC_ENCODE] unknown exception while encoding media for slot %d, task %d\n",
                        job->id_slot, job->id_task);
                res = 1;
            }

            const int64_t t_end = ggml_time_us();
            job->t_end_us.store(t_end);

            {
                std::lock_guard<std::mutex> lock(mutex);
                busy = false;
                job->state.store(res == 0 ? server_media_job::JOB_DONE : server_media_job::JOB_FAILED,
                                 std::memory_order_release);
            }
            cv_done.notify_all();

            const int64_t t_start  = job->t_start_us.load();
            const int64_t t_submit = job->t_submit_us.load();
            if (res == 0) {
                SRV_INF("[TAG_MTMD_ASYNC_ENCODE] slot %d, task %d: encoded %zu media chunk(s), %zu tokens, in %.2f s (queued %.2f s)%s\n",
                        job->id_slot, job->id_task, job->idxs.size(), job->n_tokens,
                        (t_end - t_start) / 1e6, (t_start - t_submit) / 1e6,
                        job->cancelled.load() ? ", request gone, result dropped" : "");
            } else {
                SRV_ERR("[TAG_MTMD_ASYNC_ENCODE] slot %d, task %d: failed to encode %zu media chunk(s) after %.2f s\n",
                        job->id_slot, job->id_task, job->idxs.size(), (t_end - t_start) / 1e6);
            }
        }
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // [TAG_MTMD_ASYNC_ENCODE] the encode job of the next media chunk(s) of this prompt: queued, running, or done
    // and not decoded yet. The encoder thread holds its own reference while the job is queued or running.
    std::shared_ptr<server_media_job> media_job;

    // [TAG_MTMD_ASYNC_ENCODE] the prompt stands at a media chunk whose encode has not finished
    bool media_blocked() const {
        if (!media_job || media_job->finished() || state != SLOT_STATE_PROCESSING_PROMPT || !task || parked) {
            return false;
        }
        const size_t idx = prompt.tokens.size();
        return idx < task->tokens.size() && task->tokens[idx] == LLAMA_TOKEN_NULL;
    }

    // [TAG_MTMD_ASYNC_ENCODE] a job still in the encoder is skipped or its result dropped there
    void media_job_drop() {
        if (media_job) {
            media_job->cancelled.store(true);
            media_job.reset();
        }
    }

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    std::vector<common_speculative_token_dist> spec_dists;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // [TAG_CKPT_BYTE_BUDGET] log the byte-budget cap once per slot, not once per checkpoint
    bool logged_ckpt_budget = false;

    // [TAG_CKPT_BUFFER_REUSE] the buffer of the last evicted checkpoint, handed to the next one
    std::vector<uint8_t> ckpt_spare;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    // [TAG_SHARED_PREFIX_FANOUT] When other tasks are waiting on the prefix this slot is currently
    // processing, stop filling at share_prefix_stop so one batch ends exactly on the boundary. The
    // state saved at that instant IS the shared prefix, which is the only form a follower can use:
    // the Gated DeltaNet layers cannot rewind a recurrent state to an earlier position.
    int32_t share_prefix_stop      = 0;
    bool    share_prefix_published = false;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache, const server_tokens * tokens_next = nullptr,
                     bool shared = false) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft, tokens_next, shared);
    if (cur != nullptr) {
        cur->shared = shared;
    }
        if (cur == nullptr) {
            return false;
        }

        // [TAG_POOL_PREEMPT] a short read would leave an entry whose blob does not describe its tokens
        if (llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE) != cur_size_tgt ||
            (ctx_dft && llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) != cur_size_dft)) {
            SLT_WRN(*this, "%s", "prompt save read fewer bytes than the state size, entry dropped\n");
            prompt_cache.states.pop_back();   // alloc() pushed this entry last
            return false;
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // [TAG_JARVIS_SLOTS] per-slot monitoring totals. reset() zeroes `stats` in the same main-loop step
    // that releases a request, so /slots could never show how a request ended and a poller lost the
    // tail of every request. release() folds each finished request in first, and these never reset.
    uint64_t mon_n_gen_total           = 0;
    uint64_t mon_n_prompt_total        = 0;
    int64_t  mon_t_gen_total_us        = 0;
    uint64_t mon_n_gen_steps_total     = 0;   // n_gen - 1 per request: the tokens t_gen_us() actually timed
    // the finished request's own numbers, which `stats` no longer holds after reset()
    uint64_t mon_last_n_gen            = 0;
    uint64_t mon_last_n_prompt         = 0;
    uint64_t mon_last_n_prompt_cached  = 0;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // [TAG_BS_LAZY_GRAMMAR] the target context currently holds this slot's backend sampler
    bool bs_attached = false;

    // [TAG_POOL_PREEMPT] While parked, the target memory holds nothing for seq `id`. The complete per-sequence state
    // (attention KV plus the recurrent row, flags 0) lives in this host blob, the drafter keeps its own cells in its own
    // context, and every other field (task, sampler, grammar, sampled token, draft, stats, generated text, stream) is
    // untouched until resume.
    bool                 parked          = false;
    std::vector<uint8_t> park_tgt;
    int32_t              park_cells      = 0;      // attention cells the park freed, needed again to resume
    size_t               park_n_tokens   = 0;      // prompt.tokens.size() at park, verified on resume
    slot_state           park_state      = SLOT_STATE_IDLE;
    int64_t              t_park_us       = 0;
    int64_t              t_run_us        = 0;      // launch or last resume, for victim ordering
    int32_t              n_park_fail     = 0;
    uint64_t             n_parks_total   = 0;      // never reset, /slots
    uint64_t             ckpt_build_iter = 0;      // pool iteration that created the newest context checkpoint
    std::function<void(server_slot &)> callback_on_park_drop;

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        GGML_ASSERT(!parked); // [TAG_POOL_PREEMPT] release() drops a park before it resets

        // [TAG_SHARED_PREFIX_FANOUT]
        share_prefix_stop      = 0;
        share_prefix_published = false;

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_dists.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);
        bs_attached = false;   // [TAG_BS_LAZY_GRAMMAR]

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
        media_job_drop(); // [TAG_MTMD_ASYNC_ENCODE] cancel, disconnect, error: the encode of this request is not needed
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            // [TAG_JARVIS_SLOTS] fold this request into the monitoring totals before reset() clears
            // it. A child inherits its parent's prompt stats (copy_state_to), so it adds no prompt.
            {
                const bool child = task->is_child();
                mon_n_gen_total          += stats.n_gen;
                mon_t_gen_total_us       += stats.t_gen_us();
                mon_n_gen_steps_total    += stats.n_gen_steps();
                mon_n_prompt_total       += child ? 0 : stats.n_prompt_processed;
                mon_last_n_gen            = stats.n_gen;
                mon_last_n_prompt         = child ? 0 : stats.n_prompt_processed;
                mon_last_n_prompt_cached  = stats.n_prompt_cached;
            }

            // [TAG_POOL_PREEMPT] a parked request holds no cells: its blob goes to the prompt cache so a follow-up can restore it
            if (parked) {
                callback_on_park_drop(*this);
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        // [TAG_POOL_PREEMPT]
        res["parked"]        = parked;
        res["n_parks_total"] = n_parks_total;
        res["park_cells"]    = parked ? park_cells : 0;

        // [TAG_MTMD_ASYNC_ENCODE] waiting_media: the prompt is held at a media chunk until its encode finishes
        res["waiting_media"] = media_blocked();
        if (media_job) {
            const int64_t t_now    = ggml_time_us();
            const int64_t t_submit = media_job->t_submit_us.load();
            const int64_t t_start  = media_job->t_start_us.load();
            const int64_t t_end    = media_job->t_end_us.load();
            res["media_encode"] = {
                {"state",       media_job->state_str()},
                {"n_chunks",    media_job->idxs.size()},
                {"n_tokens",    media_job->n_tokens},
                {"t_queued_ms", t_submit > 0 ? ((t_start > 0 ? t_start : t_now) - t_submit) / 1000.0 : 0.0},
                {"t_encode_ms", t_start  > 0 ? ((t_end   > 0 ? t_end   : t_now) - t_start ) / 1000.0 : 0.0},
            };
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        // [TAG_JARVIS_SLOTS] running totals: every finished request plus the live one. They only go
        // up, including across a release, because release() folds `stats` in and clears `task` in
        // the same step - so a client can diff two polls exactly.
        {
            const bool live = (bool) task;
            const bool child = live && task->is_child();
            res["n_decoded_total"]          = mon_n_gen_total + (live ? stats.n_gen : 0);
            res["n_prompt_processed_total"] = mon_n_prompt_total + (live && !child ? stats.n_prompt_processed : 0);
            res["t_gen_total_ms"]           = (double) (mon_t_gen_total_us + (live ? stats.t_gen_us() : 0)) / 1000.0;
            // the first token of each request is free (prompt logits) and untimed, so a rate is
            // n_gen_steps_total / t_gen_total_ms - the same convention as timings.predicted_per_second
            res["n_gen_steps_total"]        = mon_n_gen_steps_total + (live ? stats.n_gen_steps() : 0);
            if (!live && task_prev) {
                res["last_n_decoded"]          = mon_last_n_gen;
                res["last_n_prompt_processed"] = mon_last_n_prompt;
                res["last_n_prompt_cache"]     = mon_last_n_prompt_cached;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// [TAG_MTMD_ASYNC_ENCODE] The decode half of process_mtmd_chunk(), moved out unchanged so that the synchronous
// path and the async path (embeddings from a server_media_job) run the very same code: put the embeddings `embd`
// of the media chunk that starts at `idx` into the llama context at the slot's next position, hand the same
// batches to the drafter, and repair the drafter's positions after the chunk. Main thread only.
// returns 0 on success, -1 on error
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
static int process_mtmd_chunk_decode(const server_slot & slot, size_t idx, float * embd, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    GGML_ASSERT(embd != nullptr);
    const auto & mctx  = slot.mctx;
    const auto & chunk = slot.task->tokens.find_chunk(idx);

    // [TAG_SPEC_MEDIA_POS] see common/speculative.cpp. The drafter is text-only and now SKIPS
    // media batches, so its cache stops at med_pos_0 - 1 while the target moves on to
    // med_pos_0 + med_n_pos. Slide the drafter's sequence forward by exactly the positions the
    // media chunk consumes, so the first post-media injection is consecutive.
    //
    // pos_next() here is still the position BEFORE this chunk: push_back_placeholder() runs after
    // process_mtmd_chunk returns. n_pos is what the sequence actually advances by, max(nx, ny)
    // under M-RoPE. It is NOT n_tokens: a 20x13 image is 260 rows but only 20 positions.
    //
    // Geometrically the drafter then sees the pre-image text glued straight onto the post-image
    // text, so distances across the seam are those of the same conversation with the image
    // removed. That is the right view for a model that cannot read the image, and it keeps the
    // entire pre-image context, unlike dropping the draft cache.
    const llama_pos med_pos_0 = slot.prompt.tokens.pos_next();
    const llama_pos med_n_pos = mtmd_input_chunk_get_n_pos(chunk.get());

    auto repair_draft_after_media = [&]() {
        if (slot.ctx_dft == nullptr) {
            return;
        }

        // llama_get_memory(slot.ctx_dft), NOT slot.mem: common_memory::seq_add applies to BOTH
        // contexts, and llama_kv_cache::seq_add asserts n_pos_per_embd() == 1, which the M-RoPE
        // TARGET does not satisfy. Shifting through slot.mem would abort.
        auto * mem_dft = llama_get_memory(slot.ctx_dft);

        const llama_pos pos_dft = llama_memory_seq_pos_max(mem_dft, slot.id);
        if (pos_dft < 0) {
            // Empty draft cache, e.g. [TAG_SPEC_PREFILL_TAIL] already wiped it. An empty sequence
            // has no consecutiveness constraint, so there is nothing to repair.
            return;
        }

        // Derived from the observed pos_max rather than assumed, so a drafter that is behind for
        // any other reason still lands exactly on the position before the next text token.
        const llama_pos delta = (med_pos_0 + med_n_pos - 1) - pos_dft;
        if (delta == 0) {
            return;
        }

        if (delta > 0 && llama_memory_can_shift(mem_dft)) {
            llama_memory_seq_add(mem_dft, slot.id, -1, -1, delta);
        } else {
            // The draft K cache cannot be re-RoPEd (a turbo K cache via --spec-draft-type-k), or
            // the drafter is somehow ahead. Drop its sequence: it is all-SWA, it re-primes from
            // the following text, and an empty sequence imposes no constraint. Same remedy
            // [TAG_SPEC_PREFILL_TAIL] already uses.
            llama_memory_seq_rm(mem_dft, slot.id, -1, -1);
        }
    };

    void * cb_data = slot.spec;
    static auto cb = [](llama_batch batch, void * user_data) {
        common_speculative * spec = static_cast<common_speculative *>(user_data);
        if (!common_speculative_process(spec, batch)) {
            return 1;
        }
        return 0;
    };

    llama_pos new_n_past; // unused for now
    const int32_t res = mtmd_helper_decode_image_chunk(
        mctx,
        slot.ctx_tgt,
        chunk.get(),
        embd,
        slot.prompt.tokens.pos_next(),
        slot.id,
        llama_n_batch(slot.ctx_tgt),
        &new_n_past,
        cb,
        cb_data
    );
    if (res != 0) {
        SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
        return -1;
    }
    repair_draft_after_media();

    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
    return 0; // success
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                // [TAG_MTMD_ASYNC_ENCODE] the decode itself lives in process_mtmd_chunk_decode()
                return process_mtmd_chunk_decode(slot, idx, embd, n_tokens_out);
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    // [TAG_MTMD_ASYNC_ENCODE] true when media is encoded on media_encoder's thread, decided by media_async_init()
    bool                 media_async = false;
    server_media_encoder media_encoder;

    void destroy() {
        // [TAG_MTMD_ASYNC_ENCODE] the encoder thread uses mctx: drop the jobs and stop it before mctx is freed.
        // A job that is already running is waited for, a clip encode cannot be interrupted.
        for (auto & slot : slots) {
            slot.media_job_drop();
        }
        media_encoder.stop();
        media_async = false;

        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            // [TAG_MMPROJ_THREADS] --mmproj-threads N > 0 overrides -t for the vision encoder only
            mparams.n_threads        = params_base.mmproj_n_threads > 0 ? params_base.mmproj_n_threads : params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            media_async_init(mparams); // [TAG_MTMD_ASYNC_ENCODE]

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            // [TAG_SYNC_NCTX_NOCAP] upstream's --kv-unified-per-slot reporting is kept; its n_ctx_train cap is not (see n_ctx_slot())
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                // Upstream caps n_ctx_slot to n_ctx_train here. We deliberately do not, because a
                // caller running YaRN / linear rope scaling legitimately wants the longer window.
                //
                // But the previous version of this branch ASSERTED that scaling was configured and
                // said so in the log, without checking. If it is not configured, the model is being
                // run past its trained length with no extension at all and quality degrades silently
                // - the worst kind of failure, because the log claimed the opposite. So check.
                // Defaults are UNSPECIFIED / 0.0f / -1.0f, so "configured" means moved off those.
                // rope_scale_train != 1.0f covers a model that ships its own scaling in the GGUF.
                const float rope_scale_train = llama_model_rope_freq_scale_train(model_tgt);
                const auto  rst              = params_base.rope_scaling_type;
                const bool  scaling_active   =
                       (rst != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED && rst != LLAMA_ROPE_SCALING_TYPE_NONE)
                    || params_base.rope_freq_scale != 0.0f
                    || params_base.yarn_ext_factor >  0.0f
                    || rope_scale_train           != 1.0f;

                if (scaling_active) {
                    SRV_WRN("the slot context (%d) exceeds the training context (%d) - rope scaling is active, extending\n",
                            n_ctx_capped, n_ctx_train);
                } else {
                    SRV_WRN("the slot context (%d) exceeds the training context (%d) and NO rope scaling is configured.\n",
                            n_ctx_capped, n_ctx_train);
                    SRV_WRN("%s", "    output quality will degrade beyond the trained length. Either lower -c, or set\n");
                    SRV_WRN("%s", "    --rope-scaling yarn together with --yarn-ext-factor / --rope-scale.\n");
                }
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        {
            // [TAG_SPEC_CKPT_PROBE] Which rollback mechanism the target supports decides
            // whether every speculative step has to checkpoint the whole sequence state.
            const char * rmn = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO   ? "NO"   :
                               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ? "FULL" :
                               ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ? "PART" : "RS";
            fprintf(stderr, "turbo-probe: ctx_tgt seq_rm = %s, n_rs_seq = %d\n",
                    rmn, (int) llama_n_rs_seq(ctx_tgt));
            fflush(stderr);
        }
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.callback_on_park_drop = [this](server_slot & s) { pool_drop_park(s, /*to_cache =*/ true); }; // [TAG_POOL_PREEMPT]

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        pool_init(); // [TAG_POOL_PREEMPT]

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    // [TAG_SHARED_PREFIX_FANOUT] Should this task wait for a slot that is already processing the
    // same prefix?
    //
    // Several agent harnesses share an identical system prompt + skills + tool block - here ~16,000
    // tokens - and they arrive together. get_available_slot below can only reuse a prefix from a
    // slot that already holds tokens and is idle, so at t=0 every one of them starts from zero and
    // the same 16,000 tokens are processed once per slot. Deferring the followers lets the leader
    // finish, publish its state to the prompt cache, and have them restore it instead.
    //
    // Only waits on a leader that is STILL mid-prompt. Once the leader's prompt is complete the
    // state is in the cache and there is nothing left to wait for, so this goes false and the task
    // proceeds - which is also why deferral here cannot starve a task.
    bool share_prefix_should_defer(const server_task & task) {
        static const bool enabled = [] {
            const char * s = getenv("TURBO_SHARE_PREFIX");
            return !(s && s[0] == '0' && s[1] == '\0');
        }();
        static const int share_min = [] {
            const char * s = getenv("TURBO_SHARE_PREFIX_MIN");
            const int    v = s ? atoi(s) : 0;
            return v > 0 ? v : 2048;
        }();

        if (!enabled || !prompt_cache || slots.size() < 2) {
            return false;
        }

        if (task.type != SERVER_TASK_TYPE_COMPLETION || task.id_slot != -1) {
            return false;
        }

        for (auto & slot : slots) {
            // STARTED as well as PROCESSING_PROMPT: every task in one arrival burst is launched
            // before update_slots runs even once, so the leader is still in STARTED ("after
            // assigning a task and about to process prompt") when its followers show up. Matching
            // only PROCESSING_PROMPT meant this never fired for simultaneous requests, which is
            // precisely the case it exists for.
            const bool leader_busy_on_prompt =
                slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_PROCESSING_PROMPT;

            if (!leader_busy_on_prompt || !slot.task || slot.parked) { // [TAG_POOL_PREEMPT] a parked leader cannot publish
                continue;
            }

            // Already handed its prefix over. Without this a released follower simply re-defers on
            // whichever slot is now mid-prompt and never gets anywhere.
            if (slot.share_prefix_published) {
                continue;
            }

            const int lcp = (int) slot.task->tokens.get_common_prefix(task.tokens);
            if (lcp < share_min) {
                continue;
            }

            // Past the branch point already, so pausing there is no longer possible and waiting
            // would buy nothing.
            if ((int) slot.prompt.n_tokens() >= lcp) {
                continue;
            }

            // Pause the leader ON the boundary. If several followers share different amounts, the
            // shortest wins - that prefix is common to all of them.
            const int stop = std::min(lcp, (int) slot.task->n_tokens());
            if (slot.share_prefix_stop == 0 || stop < slot.share_prefix_stop) {
                slot.share_prefix_stop = stop;
            }

            SRV_INF("share_prefix: task %d shares %d tokens with slot %d - waiting for it to publish "
                    "that prefix instead of processing it again\n", task.id, lcp, slot.id);
            return true;
        }

        return false;
    }

    // [TAG_SHARED_PREFIX_FANOUT] The leader finished its prompt: publish the state so the tasks
    // deferred above can restore it, then release them. Called once, from the point where prompt
    // processing completes.
    // [TAG_SHARED_PREFIX_FANOUT] Run after each decode: any slot that has just landed on its
    // shared boundary publishes the state there and wakes the tasks waiting for it.
    //
    // The boundary matters. Saving once the leader's whole prompt is done produces a state at the
    // END of that prompt, and a follower would have to rewind it to the branch point - impossible
    // for the 48 recurrent layers, so the server falls back to a full re-prefill and the whole
    // exercise is wasted. Saved ON the boundary, the cached prompt IS the shared prefix, the
    // follower's f_keep is 1.0, and it processes only its own tail.
    void share_prefix_publish_ready() {
        if (!prompt_cache || slots.size() < 2) {
            return;
        }

        for (auto & slot : slots) {
            if (slot.share_prefix_stop == 0 || slot.parked) {
                continue;
            }

            if ((int) slot.prompt.n_tokens() < slot.share_prefix_stop) {
                continue; // not on the boundary yet
            }

            const int n_shared = slot.share_prefix_stop;

            slot.share_prefix_stop      = 0;
            slot.share_prefix_published = true;

            // shared = true: this is a boundary entry, exempt from containment
            if (slot.prompt_save(*prompt_cache, nullptr, /*shared =*/ true)) {
                prompt_cache->update();
                SLT_INF(slot, "share_prefix: published %d tokens at the shared boundary\n", n_shared);
            } else {
                SLT_WRN(slot, "share_prefix: could not publish %d tokens - followers will re-process\n",
                        n_shared);
            }

            // wake everything parked on this prefix
            for (size_t i = 0; i < slots.size(); ++i) {
                queue_tasks.pop_deferred_task(-1);
            }
        }
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                // [TAG_FKEEP_NAN] compare with the slot get_slot_by_id returned, not the raw id: it wraps an
                // out-of-range id_slot (N+k -> k), and the raw compare then skipped every slot
                if (task.id_slot != -1 && &slot != ret) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                // [TAG_FKEEP_NAN] an empty prompt reaches here (it is rejected later, in pre_decode): 0, not 0/0
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = task.tokens.empty() ? 0.0f : float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                // [TAG_FKEEP_NAN] With id_slot set, ret is the requested slot even when the loop above skipped it for
                // being empty (never used, or cleared by --cache-idle-slots). Its size is 0, so f_keep was 0/0 = NaN,
                // "f_keep < 0.5f" was false, update_cache stayed false and the prompt cache was never searched: the
                // whole prompt was processed again. An empty slot keeps nothing, so f_keep = 0 and the cache is
                // searched and loaded exactly as on the id_slot = -1 path, where LRU picks an empty slot.
                const size_t n_cur  = ret->prompt.tokens.size();
                const float  f_keep = n_cur > 0 ? (f_sim_best*task.tokens.size()) / n_cur : 0.0f;

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // [TAG_FKEEP_NAN] The same for --slot-prompt-similarity 0, where the block above does not run: a requested
        // slot that holds nothing has nothing to lose, so look for the prompt in the cache.
        if (ret != nullptr && task.id_slot != -1 && ret->prompt.tokens.empty()) {
            update_cache = true;
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            // [TAG_POOL_PREEMPT] a requested slot that is busy is deferred by the caller, never save or load over its live memory
            if (ret->is_processing()) {
                return ret;
            }

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                // [TAG_POOL_PREEMPT] A cached state that does not fit the unified pool fails to restore after this slot's
                // own cells are already gone, so the whole prompt is processed again, and a long prompt holds the batch
                // until it is done. Measured with 4 x 68K follow-ups: one agent processed 68,037 tokens again for 76 s
                // while two others waited 75 s behind it. Room held by idle slots is taken first (saved, nothing is
                // lost). Room held by busy slots is waited for: the task is deferred, retried whenever a slot releases,
                // and after TURBO_POOL_LOAD_WAIT_MS (default 60000) it falls back to processing the prompt.
                if (pool_preempt && params_base.kv_unified) {
                    const auto it = prompt_cache->find(ret->prompt, task.tokens);
                    if (it != prompt_cache->states.end()) {
                        static const int64_t wait_max_us = [] {
                            const char * e = getenv("TURBO_POOL_LOAD_WAIT_MS");
                            return (e ? (int64_t) atoll(e) : (int64_t) 60000) * 1000;
                        }();
                        const int64_t need = (int64_t) it->prompt.tokens.size();
                        const auto    room = [&]() {
                            return (int64_t) llama_memory_attn_n_free_ext(ctx_tgt, 0) + (int64_t) ret->prompt.tokens.size();
                        };
                        int64_t idle = 0;
                        for (const auto & o : slots) {
                            if (&o != ret && !o.is_processing()) {
                                idle += (int64_t) o.prompt.tokens.size();
                            }
                        }
                        if (room() + idle < need) {
                            if (pool_load_wait_t0.size() > 256) {
                                pool_load_wait_t0.clear();   // tasks cancelled while deferred
                            }
                            const int64_t now = ggml_time_us();
                            const auto    w   = pool_load_wait_t0.emplace(task.id, now).first;
                            if (now - w->second < wait_max_us) {
                                SLT_WRN(*ret, "__TEST_TAG_POOL_LOAD_WAIT__ task %d: cached prompt of %" PRId64 " cells, %" PRId64 " free with this slot, %" PRId64 " in idle slots, waited %.1f s\n",
                                        task.id, need, room(), idle, (now - w->second) / 1e6);
                                return nullptr;
                            }
                            SLT_WRN(*ret, "__TEST_TAG_POOL_LOAD_GIVEUP__ task %d waited %.1f s, processing the prompt instead\n",
                                    task.id, (now - w->second) / 1e6);
                        }
                        for (auto & o : slots) {
                            if (room() >= need) {
                                break;
                            }
                            if (&o == ret || o.is_processing() || o.prompt.tokens.empty()) {
                                continue;
                            }
                            SLT_INF(o, "__TEST_TAG_POOL_LOAD_ROOM__ saving and clearing %zu tokens for a cached prompt of %" PRId64 " cells\n",
                                    o.prompt.tokens.size(), need);
                            o.prompt_save(*prompt_cache, &task.tokens);
                            o.prompt_clear();
                        }
                    }
                    pool_load_wait_t0.erase(task.id);
                }

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache, &task.tokens);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // [TAG_POOL_PREEMPT] with the pool policy on, an idle slot is saved to the prompt cache before it is purged, and a
    // slot that still has rows in the undecoded part of the batch (at or after `off`) is never purged.
    bool try_clear_idle_slots(int32_t off = -1) {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing() || slot.prompt.n_tokens() == 0) {
                continue;
            }

            if (pool_preempt) {
                if (off >= 0 && batch_has_rows(slot.id, off)) {
                    continue;
                }
                if (prompt_cache && slot.prompt_save(*prompt_cache)) {
                    prompt_cache->update();
                    SLT_INF(slot, "__TEST_TAG_POOL_IDLE_SAVED__ %zu tokens\n", slot.prompt.tokens.size());
                } else {
                    SLT_WRN(slot, "__TEST_TAG_POOL_IDLE_UNSAVED__ %zu tokens (already cached, over the cache limit, or no cache)\n",
                            slot.prompt.tokens.size());
                }
            }

            SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

            slot.prompt_clear();

            res = true;

            // clear slots one by one
            break;
        }

        return res;
    }

    //
    // [TAG_POOL_PREEMPT] unified KV pool exhaustion
    //
    // --kv-unified gives every slot the whole 262,144-cell pool, so four agents can outgrow it together while each is
    // far below its own n_ctx. The upstream reaction to llama_decode returning 1 was to halve n_batch down to 1 and
    // then fail EVERY processing slot. Halving also cut speculative verify groups in two, and post_decode then threw for
    // each slot whose verify rows fell outside the view. Reproduced: 4 x 62K prompts + 6,000 tokens each, 2 of 4 agents
    // got HTTP 500 with DFlash2, all 4 without speculation.
    //
    // Invariants this relies on. A llama_decode view that returns 1 applies nothing to memory. Generating slots put
    // their sampled + draft rows first, [0, n_gen_rows), so a view that starts at 0 and is at least n_gen_rows long
    // keeps every verify group whole. At the top of an iteration a slot that is not parked has
    // prompt.tokens.pos_next() == seq_pos_max + 1, no i_batch and no verify rows, which is the state a park copies.
    //
    bool     pool_preempt          = false;
    bool     pool_exhausted        = false;
    bool     pool_handled_iter     = false;
    int32_t  pool_park_margin      = 8192;
    int64_t  pool_min_run_us       = 20000000;
    std::map<int, int64_t> pool_load_wait_t0;   // task id -> first time its cached prompt did not fit the pool
    int64_t  pool_dbg_hold_us      = 0;
    uint64_t pool_iter             = 0;
    int      pool_n_parked         = 0;
    int      pool_no_progress      = 0;
    int      pool_want_room        = -1;       // slot whose next image does not fit, set by the prompt pass
    struct pool_dbg_trigger { bool gen; int64_t at; bool nopark; bool done; };
    std::vector<pool_dbg_trigger> pool_dbg;    // TURBO_POOL_DEBUG_FAIL
    int      pool_dbg_firing       = -1;
    int32_t  pool_dbg_firing_off   = -1;
    int64_t  pool_dbg_prompt_views = 0;

    void pool_init() {
        auto env_i64 = [](const char * n, int64_t d) {
            const char * e = getenv(n);
            return e ? (int64_t) atoll(e) : d;
        };
        const int32_t n_gen_max = params_base.n_parallel * (1 + (spec ? common_speculative_n_max(spec.get()) : 0));
        const int32_t n_free    = llama_memory_attn_n_free_ext(ctx_tgt, 0);
        pool_preempt = params_base.kv_unified && params_base.n_parallel > 1
                    && env_i64("TURBO_POOL_PREEMPT", 1) != 0
                    && n_free >= 0 && n_gen_max <= llama_n_batch(ctx_tgt);
        pool_park_margin = (int32_t) env_i64("TURBO_POOL_PARK_MARGIN", 8192);
        pool_min_run_us  = 1000 * env_i64("TURBO_POOL_MIN_RUN_MS", 20000);
        pool_dbg_hold_us = 1000 * env_i64("TURBO_POOL_DEBUG_HOLD_MS", 0);
        if (pool_preempt) {
            pool_dbg_parse(getenv("TURBO_POOL_DEBUG_FAIL"));
        }
        SRV_WRN("[TAG_POOL_PREEMPT] %s: park margin %d cells, min run %lld ms, n_free %d, gen rows max %d, debug triggers %zu\n",
                pool_preempt ? "on" : "off", pool_park_margin, (long long) (pool_min_run_us / 1000), n_free, n_gen_max,
                pool_dbg.size());
    }

    void pool_dbg_parse(const char * s) {
        pool_dbg.clear();
        if (s == nullptr) {
            return;
        }
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            pool_dbg_trigger t = { true, 0, false, false };
            if (item.rfind("gen@", 0) == 0) {
                t.gen = true;
                item  = item.substr(4);
            } else if (item.rfind("prompt@", 0) == 0) {
                t.gen = false;
                item  = item.substr(7);
            } else {
                SRV_WRN("[TAG_POOL_PREEMPT] ignoring debug trigger '%s'\n", item.c_str());
                continue;
            }
            const size_t c = item.find(':');
            if (c != std::string::npos) {
                t.nopark = item.substr(c + 1) == "nopark";
                item     = item.substr(0, c);
            }
            t.at = atoll(item.c_str());
            pool_dbg.push_back(t);
        }
    }

    // test hook, runs inside the decode yield: pretend the view at `off` found no room, once per trigger, in order
    bool pool_dbg_fire(int32_t off) {
        if (pool_dbg.empty()) {
            return false;
        }
        if (pool_dbg_firing >= 0) {
            return pool_dbg_firing_off == off;
        }
        for (size_t i = 0; i < pool_dbg.size(); ++i) {
            auto & t = pool_dbg[i];
            if (t.done) {
                continue;
            }
            bool hit = false;
            if (t.gen) {
                if (off == 0 && batch.n_gen_rows > 0) {
                    for (const auto & s : slots) {
                        if (s.state == SLOT_STATE_GENERATING && !s.parked && (int64_t) s.stats.n_gen >= t.at) {
                            hit = true;
                            break;
                        }
                    }
                }
            } else {
                hit = off >= batch.n_gen_rows && pool_dbg_prompt_views >= t.at;
            }
            if (hit) {
                t.done              = true;
                pool_dbg_firing     = (int) i;
                pool_dbg_firing_off = off;
                return true;
            }
            break;
        }
        return false;
    }

    // ends a firing trigger, returns true when it asked for the undo only
    bool pool_dbg_consume() {
        if (pool_dbg_firing < 0) {
            return false;
        }
        const bool nopark = pool_dbg[pool_dbg_firing].nopark;
        pool_dbg_firing     = -1;
        pool_dbg_firing_off = -1;
        return nopark;
    }

    bool batch_has_rows(int id_slot, int32_t off) const {
        for (int32_t i = std::max(0, off); i < batch.size(); ++i) {
            if (batch.tokens[i].id_slot == id_slot) {
                return true;
            }
        }
        return false;
    }

    // Host-side undo of the rows at [off, size) that were never decoded. Each slot's rows are the last tokens it pushed
    // this iteration, one contiguous run, so dropping that many tokens restores the state before the batch was built.
    // Memory needs no undo: the failed view applied nothing. Drafts are kept and re-verified next iteration.
    void pool_abandon_tail(int32_t off) {
        std::vector<int32_t> k_rows(slots.size(), 0);
        for (int32_t i = off; i < batch.size(); ++i) {
            k_rows[batch.tokens[i].id_slot]++;
        }
        for (auto & slot : slots) {
            const int32_t k = k_rows[slot.id];
            if (k == 0 && slot.i_batch < off) {
                continue;
            }
            if (k > 0) {
                GGML_ASSERT((size_t) k <= slot.prompt.tokens.size());
                slot.prompt.tokens.keep_first(slot.prompt.tokens.size() - k);
            }
            if (!slot.spec_i_batch.empty()) {
                GGML_ASSERT(off == 0);
                slot.spec_i_batch.clear();
            }
            if (slot.i_batch >= off) {
                slot.i_batch = -1;
            }
            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                slot.state = SLOT_STATE_PROCESSING_PROMPT;
            }
            if (slot.ckpt_build_iter == pool_iter && slot.state == SLOT_STATE_PROCESSING_PROMPT &&
                !slot.prompt.checkpoints.empty() &&
                slot.prompt.checkpoints.back().n_tokens == (int64_t) slot.prompt.n_tokens()) {
                if (slot.ckpt_spare.empty()) {
                    slot.ckpt_spare = std::move(slot.prompt.checkpoints.back().data_tgt);
                }
                slot.prompt.checkpoints.pop_back();   // taken for rows that were never decoded
            }
        }
    }

    // slots that can be parked, best victim first: running longer than the minimum, not a shared-prefix leader, most cells
    std::vector<server_slot *> pool_victims(int64_t now, const server_slot * exclude) {
        std::vector<server_slot *> v;
        for (auto & s : slots) {
            if (&s == exclude || !s.is_processing() || s.parked || !s.task) {
                continue;
            }
            if (s.state != SLOT_STATE_GENERATING && s.state != SLOT_STATE_PROCESSING_PROMPT) {
                continue;
            }
            if (s.task->type != SERVER_TASK_TYPE_COMPLETION && s.task->type != SERVER_TASK_TYPE_INFILL) {
                continue;
            }
            if (s.task->is_parent() || s.task->is_child() || s.prompt.n_tokens() == 0) {
                continue;
            }
            v.push_back(&s);
        }
        std::stable_sort(v.begin(), v.end(), [&](const server_slot * a, const server_slot * b) {
            const bool ra = now - a->t_run_us >= pool_min_run_us;
            const bool rb = now - b->t_run_us >= pool_min_run_us;
            if (ra != rb) {
                return ra;
            }
            const bool la = a->share_prefix_stop != 0;
            const bool lb = b->share_prefix_stop != 0;
            if (la != lb) {
                return lb;
            }
            return a->prompt.tokens.size() > b->prompt.tokens.size();
        });
        return v;
    }

    // Last resort: ONE request fails, saved to the prompt cache first, with the wording pi treats as a context overflow
    // (it compacts instead of dropping the turn). An n_cmpl family fails together so no child waits forever.
    void pool_fail_one(const char * why) {
        server_slot * v = nullptr;
        for (auto * s : pool_victims(ggml_time_us(), nullptr)) {
            if (v == nullptr || s->prompt.tokens.size() > v->prompt.tokens.size()) {
                v = s;
            }
        }
        std::vector<server_slot *> group;
        if (v != nullptr) {
            group.push_back(v);
        } else {
            for (auto & s : slots) {
                if (s.is_processing() && !s.parked && s.task && s.task->n_tokens() > 0 &&
                    (v == nullptr || s.prompt.tokens.size() > v->prompt.tokens.size())) {
                    v = &s;
                }
            }
            if (v == nullptr) {
                return;
            }
            const int id_family = v->task->is_child() ? v->task->id_parent : v->task->id;
            for (auto & s : slots) {
                if (s.task && !s.parked && (s.task->id == id_family || s.task->id_parent == id_family)) {
                    group.push_back(&s);
                }
            }
        }
        for (auto * s : group) {
            if (prompt_cache && s->state != SLOT_STATE_WAIT_OTHER && s->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }
            SLT_ERR(*s, "__TEST_TAG_POOL_FAIL_ONE__ %s\n", why);
            send_error(*s, string_format("request (%d tokens) exceeds the available context size (%d tokens) while other requests share the KV pool (%s)",
                                         (int) s->prompt.tokens.size(), s->n_ctx, why), ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            s->release();
            s->prompt_clear();
        }
    }

    // the batch tail at `off` did not fit: undo it, then free cells without losing any work
    void pool_handle_exhausted(int32_t off, const server_slot * exclude) {
        const bool    dbg_nopark = pool_dbg_consume();
        const int32_t n_free0    = llama_memory_attn_n_free_ext(ctx_tgt, 0);

        pool_abandon_tail(off);
        pool_handled_iter = true;

        SRV_WRN("__TEST_TAG_POOL_FULL__ off = %d, n_gen_rows = %d, batch = %d, n_free = %d, parked = %d\n",
                off, batch.n_gen_rows, batch.size(), n_free0, pool_n_parked);

        if (dbg_nopark) {
            return;
        }
        if (try_clear_idle_slots()) {
            return;
        }
        if (++pool_no_progress > (int) slots.size() + 2) {
            pool_no_progress = 0;
            pool_fail_one("no slot could make progress");
            return;
        }
        for (server_slot * v : pool_victims(ggml_time_us(), exclude)) {
            if (pool_park(*v, "pool full")) {
                return;
            }
        }
        pool_fail_one("no slot could be parked");
    }

    bool pool_park(server_slot & s, const char * why) {
        const llama_memory_t mt = llama_get_memory(ctx_tgt);
        const llama_pos pos_max = llama_memory_seq_pos_max(mt, s.id);
        if (s.parked || s.i_batch != -1 || !s.spec_i_batch.empty() || pos_max + 1 != s.prompt.tokens.pos_next()) {
            SLT_WRN(s, "[TAG_POOL_PREEMPT] not parking (%s): i_batch = %d, verify rows = %zu, pos_max = %d, pos_next = %d\n",
                    why, s.i_batch, s.spec_i_batch.size(), (int) pos_max, (int) s.prompt.tokens.pos_next());
            return false;
        }

        // Only the target's cells are the contended pool. The drafter holds a fixed window per sequence in its own
        // context (12.9 MiB at 20K and at 66K tokens), so it stays where it is. Sending it through a blob as well kept
        // greedy tokens identical but changed the drafts a few steps after the resume, and a seeded request diverged
        // from the same request without a park (token 40 of 3000, reproducible). Without speculation the target round
        // trip alone was identical.
        const int64_t t0    = ggml_time_us();
        const size_t  n_tgt = llama_state_seq_get_size_ext(ctx_tgt, s.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n_tgt == 0) {
            return false;
        }

        // live work outranks cached prompts: evict the oldest unshared entries until the park fits the RAM budget
        if (prompt_cache && prompt_cache->limit_size > 0) {
            for (auto it = prompt_cache->states.begin(); it != prompt_cache->states.end() &&
                 prompt_cache->size() + prompt_cache->reserved + n_tgt > prompt_cache->limit_size;) {
                if (it->shared) {
                    ++it;
                    continue;
                }
                it = prompt_cache->states.erase(it);
            }
        }

        std::vector<uint8_t> bt;
        for (;;) {
            try {
                bt.resize(n_tgt);
                break;
            } catch (const std::bad_alloc &) {
                bt = std::vector<uint8_t>();
                if (!prompt_cache || prompt_cache->states.empty()) {
                    SLT_ERR(s, "[TAG_POOL_PREEMPT] not parking (%s): cannot allocate %.1f MiB of host memory\n",
                            why, n_tgt / 1048576.0);
                    return false;
                }
                prompt_cache->states.pop_front();
            }
        }

        size_t got_tgt = 0;
        queue_tasks.yield_to_queue([&]() {
            got_tgt = llama_state_seq_get_data_ext(ctx_tgt, bt.data(), n_tgt, s.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        });
        if (got_tgt != n_tgt) {
            SLT_ERR(s, "[TAG_POOL_PREEMPT] not parking (%s): state read %zu of %zu bytes\n", why, got_tgt, n_tgt);
            return false;
        }
        const int32_t free0 = llama_memory_attn_n_free_ext(ctx_tgt, s.id);
        llama_memory_seq_rm(mt, s.id, -1, -1);   // target only
        const int32_t free1 = llama_memory_attn_n_free_ext(ctx_tgt, s.id);

        s.park_tgt      = std::move(bt);
        s.park_cells    = free1 - free0;
        s.park_n_tokens = s.prompt.tokens.size();
        s.park_state    = s.state;
        s.parked        = true;
        s.t_park_us     = ggml_time_us();
        s.n_park_fail   = 0;
        s.n_parks_total++;
        pool_n_parked++;

        if (prompt_cache) {
            prompt_cache->reserved += s.park_tgt.size();
            prompt_cache->update();
        }

        // followers waiting on this slot's prefix must not wait for the whole park
        if (s.share_prefix_stop != 0) {
            s.share_prefix_stop      = 0;
            s.share_prefix_published = true;
            for (size_t i = 0; i < slots.size(); ++i) {
                queue_tasks.pop_deferred_task(-1);
            }
        }

        SLT_WRN(s, "__TEST_TAG_POOL_PARK__ %s: state = %d, n_tokens = %zu, cells = %d, tgt = %.1f MiB, n_free %d -> %d, took %.1f ms\n",
                why, (int) s.park_state, s.park_n_tokens, s.park_cells, s.park_tgt.size() / 1048576.0,
                free0, free1, (ggml_time_us() - t0) / 1000.0);
        return true;
    }

    bool pool_restore(server_slot & s) {
        const int64_t t0 = ggml_time_us();
        const llama_memory_t mt = llama_get_memory(ctx_tgt);
        size_t put_tgt = 0;
        queue_tasks.yield_to_queue([&]() {
            put_tgt = llama_state_seq_set_data_ext(ctx_tgt, s.park_tgt.data(), s.park_tgt.size(), s.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        });
        if (put_tgt != s.park_tgt.size()) {
            llama_memory_seq_rm(mt, s.id, -1, -1);   // the blob is a const source: intact, the slot stays parked
            s.n_park_fail++;
            SLT_WRN(s, "[TAG_POOL_PREEMPT] resume failed (%zu of %zu bytes), try %d\n", put_tgt, s.park_tgt.size(), s.n_park_fail);
            return false;
        }
        if (llama_memory_seq_pos_max(mt, s.id) + 1 != s.prompt.tokens.pos_next() ||
            s.prompt.tokens.size() != s.park_n_tokens) {
            llama_memory_seq_rm(mt, s.id, -1, -1);
            s.n_park_fail++;
            SLT_ERR(s, "[TAG_POOL_PREEMPT] resumed state does not match its tokens, try %d\n", s.n_park_fail);
            return false;
        }

        const int64_t now   = ggml_time_us();
        const int64_t pause = now - s.t_park_us;
        // excluded from the durations, and shifted so /slots running totals never step backwards at the resume
        if (s.park_state == SLOT_STATE_GENERATING) {
            s.stats.t_paused_gen_us += pause;
            if (s.stats.t_gen_last != 0) {
                s.stats.t_gen_last += pause;
            }
        } else if (s.park_state == SLOT_STATE_PROCESSING_PROMPT) {
            s.stats.t_paused_prompt_us += pause;
            if (s.stats.t_prompt_last != 0) {
                s.stats.t_prompt_last += pause;
            }
        }

        if (prompt_cache) {
            prompt_cache->reserved -= std::min(prompt_cache->reserved, s.park_tgt.size());
        }
        s.park_tgt = std::vector<uint8_t>();
        s.parked   = false;
        s.t_run_us = now;
        pool_n_parked--;

        SLT_WRN(s, "__TEST_TAG_POOL_RESUME__ after %.1f s, n_tokens = %d, took %.1f ms\n",
                pause / 1e6, s.prompt.n_tokens(), (now - t0) / 1000.0);
        return true;
    }

    // a parked request is released (cancel, disconnect, abort): keep its work in the prompt cache for a follow-up
    void pool_drop_park(server_slot & s, bool to_cache) {
        if (!s.parked) {
            return;
        }
        if (prompt_cache) {
            prompt_cache->reserved -= std::min(prompt_cache->reserved, s.park_tgt.size());
            if (to_cache && !s.prompt.tokens.empty()) {
                // the drafter's cells never left its context, so its part of the entry is read now
                const size_t n_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, s.id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
                if (auto * cur = prompt_cache->alloc(s.prompt, 0, n_dft)) {
                    cur->data.main = std::move(s.park_tgt);
                    if (n_dft > 0 && llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), n_dft, s.id, LLAMA_STATE_SEQ_FLAGS_NONE) != n_dft) {
                        cur->data.drft = std::vector<uint8_t>();   // restored later with an empty drafter window: acceptance only
                    }
                }
            }
            prompt_cache->update();
        }
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), s.id, -1, -1);
        }
        SLT_WRN(s, "__TEST_TAG_POOL_PARK_DROP__ to_cache = %d, n_tokens = %zu\n", (int) to_cache, s.prompt.tokens.size());
        s.park_tgt = std::vector<uint8_t>();
        s.parked   = false;
        pool_n_parked--;
        s.prompt.clear();
    }

    void pool_restore_failed(server_slot & s) {
        SLT_ERR(s, "[TAG_POOL_PREEMPT] parked state could not be restored after %d tries\n", s.n_park_fail);
        pool_drop_park(s, false);
        send_error(s, "parked state could not be restored", ERROR_TYPE_SERVER);
        s.release();
    }

    // resume parked slots, oldest park first, never past the head
    void pool_unpark() {
        const int64_t now = ggml_time_us();
        std::vector<server_slot *> q;
        int n_active = 0;
        for (auto & s : slots) {
            if (s.parked) {
                q.push_back(&s);
            } else if (s.is_processing()) {
                n_active++;
            }
        }
        std::sort(q.begin(), q.end(), [](const server_slot * a, const server_slot * b) { return a->t_park_us < b->t_park_us; });

        for (server_slot * s : q) {
            if (now - s->t_park_us < pool_dbg_hold_us) {
                break;
            }
            const int64_t need     = (int64_t) s->park_cells + (n_active > 0 ? pool_park_margin : 0);
            int64_t       free_now = llama_memory_attn_n_free_ext(ctx_tgt, 0);
            if (free_now < need) {
                int64_t idle = 0;
                for (auto & o : slots) {
                    if (!o.is_processing()) {
                        idle += (int64_t) o.prompt.tokens.size();
                    }
                }
                if (free_now + idle >= need) {
                    while (llama_memory_attn_n_free_ext(ctx_tgt, 0) < need && try_clear_idle_slots()) {}
                    free_now = llama_memory_attn_n_free_ext(ctx_tgt, 0);
                }
            }
            if (free_now < need) {
                break;
            }
            if (!pool_restore(*s)) {
                if (s->n_park_fail >= 3) {
                    pool_restore_failed(*s);
                }
                break;
            }
            n_active++;
        }
    }

    //
    // [TAG_MTMD_ASYNC_ENCODE] asynchronous media encoding, see server_media_encoder
    //

    // Decides whether media is encoded on its own thread. Called once mctx is loaded, again after a sleep.
    void media_async_init(const mtmd_context_params & mparams) {
        media_async = false;
        if (mctx == nullptr) {
            return;
        }

        // MTMD_ASYNC_ENCODE: 0 = always synchronous, 1 = asynchronous on any device but CUDA, unset or auto = automatic
        int mode = -1;
        if (const char * e = getenv("MTMD_ASYNC_ENCODE")) {
            if (strcmp(e, "0") == 0) {
                mode = 0;
            } else if (strcmp(e, "1") == 0) {
                mode = 1;
            } else if (e[0] != '\0' && strcmp(e, "auto") != 0) {
                SRV_WRN("[TAG_MTMD_ASYNC_ENCODE] ignoring MTMD_ASYNC_ENCODE=%s, expected 0, 1 or auto\n", e);
            }
        }

        // the device clip_ctx::clip_ctx() (tools/mtmd/clip.cpp) runs the projector on, for these params
        ggml_backend_dev_t dev = nullptr;
        if (mparams.use_gpu) {
            dev = mparams.device;
            if (dev == nullptr) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            }
            if (dev == nullptr) {
                dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
            }
        }
        const bool        on_cpu   = dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const std::string dev_name = on_cpu ? std::string("CPU") : std::string(ggml_backend_dev_name(dev));

        bool is_cuda = false; // CUDA, or the same backend built for HIP (ROCm) or MUSA
        bool shared  = false; // the text model or the drafter runs on this device too
        if (on_cpu) {
            // [TAG_MTMD_ASYNC_ENCODE_CPU_TEXT] A text model on the CPU (a CPU-only build, --device none, -ngl 0) shares
            // the cores with a CPU projector: two ggml thread pools of -t / --mmproj-threads threads each would spin
            // against each other in their barriers, so it stays synchronous unless MTMD_ASYNC_ENCODE=1.
            shared = model_tgt == nullptr || llama_model_n_devices(model_tgt) == 0 || params_base.n_gpu_layers == 0;
        } else {
            ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
            const char *       reg_name = reg ? ggml_backend_reg_name(reg) : "";
            is_cuda = strcmp(reg_name, "CUDA") == 0 || strcmp(reg_name, "ROCm") == 0 || strcmp(reg_name, "MUSA") == 0;

            const llama_model * models[] = { model_tgt, model_dft };
            for (const llama_model * m : models) {
                if (m == nullptr) {
                    continue;
                }
                for (int32_t i = 0; i < llama_model_n_devices(m); ++i) {
                    shared = shared || llama_model_get_device(m, i) == dev;
                }
            }
        }

        const char * why_sync = nullptr;
        if (mode == 0) {
            why_sync = "MTMD_ASYNC_ENCODE=0";
        } else if (is_cuda) {
            why_sync = "it is fast on a CUDA device, and a second thread driving CUDA is avoided";
        } else if (shared && mode != 1) {
            why_sync = "the text model uses the same device, MTMD_ASYNC_ENCODE=1 overrides";
        }

        if (why_sync == nullptr) {
            try {
                media_encoder.start();
                media_async = true;
            } catch (const std::exception & e) {
                SRV_WRN("[TAG_MTMD_ASYNC_ENCODE] cannot start the encoder thread (%s), media is encoded synchronously\n", e.what());
                return;
            }
            SRV_INF("[TAG_MTMD_ASYNC_ENCODE] media is encoded on its own thread (projector on %s): the other slots keep "
                    "generating while an image or audio is encoded. MTMD_ASYNC_ENCODE=0 encodes synchronously\n", dev_name.c_str());
        } else {
            SRV_INF("[TAG_MTMD_ASYNC_ENCODE] media is encoded synchronously (projector on %s): %s\n", dev_name.c_str(), why_sync);
        }
    }

    // Queues the encode of the media chunk that starts at `idx`, batched with the media chunks after it by the same
    // rule as process_mtmd_chunk(), so the encoder computes exactly what the synchronous path computes. The job
    // gets its own copies of those chunks. Returns false if the job cannot be built.
    bool media_async_submit(server_slot & slot, size_t idx) {
        const auto & input_tokens = slot.task->tokens;

        // pick the chunks on the originals first, so that only the chunks that go into the batch are copied
        std::vector<std::pair<size_t, const mtmd_input_chunk *>> picked;
        {
            mtmd::batch_ptr probe(mtmd_batch_init(mctx));
            const mtmd_input_chunk * first = input_tokens.find_chunk(idx).get();
            if (mtmd_batch_add_chunk(probe.get(), first) != 0) {
                SLT_ERR(slot, "[TAG_MTMD_ASYNC_ENCODE] cannot encode the media chunk at idx = %zu\n", idx);
                return false;
            }
            picked.emplace_back(idx, first);

            size_t idx_cur = idx;
            while (true) {
                const auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
                if (next_chunk == nullptr || mtmd_batch_add_chunk(probe.get(), next_chunk->get()) != 0) {
                    break; // no more media, the batch is full, or the chunk cannot be batched with the others
                }
                picked.emplace_back(next_idx, next_chunk->get());
                idx_cur = next_idx;
            }
        }

        auto job = std::make_shared<server_media_job>();
        job->id_slot = slot.id;
        job->id_task = slot.task->id;
        job->mbatch.reset(mtmd_batch_init(mctx));

        for (const auto & [at, src] : picked) {
            mtmd::input_chunk_ptr cpy(mtmd_input_chunk_copy(src));
            if (!cpy || mtmd_batch_add_chunk(job->mbatch.get(), cpy.get()) != 0) {
                SLT_ERR(slot, "[TAG_MTMD_ASYNC_ENCODE] cannot put a copy of the media chunk at idx = %zu into an encode job\n", at);
                return false;
            }
            job->n_tokens += mtmd_input_chunk_get_n_tokens(cpy.get());
            job->idxs.push_back(at);
            job->chunks.push_back(std::move(cpy));
        }

        SLT_INF(slot, "[TAG_MTMD_ASYNC_ENCODE] queued the encode of %zu media chunk(s), %zu tokens, from idx = %zu%s, %d job(s) ahead\n",
                job->idxs.size(), job->n_tokens, idx, idx > slot.prompt.tokens.size() ? " (ahead of the prompt)" : "",
                media_encoder.n_jobs());

        slot.media_job = job;
        media_encoder.submit(std::move(job));
        return true;
    }

    // Where the slot stands with its media, called while its prompt is processed. Returns
    //    2: the prompt is at a media chunk whose encode was queued by this call: add nothing for this slot
    //    1: the prompt is at a media chunk that is still being encoded: add nothing for this slot in this iteration
    //    0: go on: the next token is text, or the embeddings of the media chunk at the prompt are ready
    //   -1: the encode failed or could not be queued: fail the request
    // While the next token is text, the next media chunk of the prompt is queued already, so that its encode overlaps
    // with the text before it instead of starting only when the prompt gets there.
    int media_async_step(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const size_t idx          = slot.prompt.tokens.size();

        // a job of another task, or one the prompt is past the last chunk of (all of it decoded), is not needed
        if (slot.media_job && (slot.media_job->id_task != slot.task->id || slot.media_job->idxs.back() < idx)) {
            slot.media_job_drop();
        }

        if (idx >= input_tokens.size()) {
            return 0;
        }

        const bool at_media = input_tokens[idx] == LLAMA_TOKEN_NULL;

        if (at_media && slot.media_job && !slot.media_job->holds(idx)) {
            // the job holds later chunks only - not expected, chunks are queued in prompt order
            slot.media_job_drop();
        }

        bool queued_now = false;
        if (!slot.media_job) {
            size_t idx_media = idx;
            if (!at_media) {
                const auto next = input_tokens.find_next_media_chunk(idx);
                if (next.first == nullptr) {
                    return 0; // no media left in this prompt
                }
                idx_media = next.second;
            }
            // [TAG_MTMD_ENCODE_CATCH] find_chunk() throws on an index that is not a chunk start, and the copies can
            // run out of memory. The synchronous path runs this inside the yield's try/catch and fails only the
            // request; here an exception would leave pre_decode() and abort every slot.
            bool ok = false;
            try {
                ok = media_async_submit(slot, idx_media);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "[TAG_MTMD_ASYNC_ENCODE] exception while queueing the media chunk at idx = %zu: %s\n", idx_media, e.what());
                ok = false;
            }
            if (!ok) {
                slot.media_job_drop();
                return -1;
            }
            queued_now = true;
        }

        if (!at_media) {
            return 0;
        }

        switch (slot.media_job->state.load(std::memory_order_acquire)) {
            case server_media_job::JOB_DONE:
                return 0;
            case server_media_job::JOB_FAILED:
            case server_media_job::JOB_CANCELLED:
                slot.media_job_drop();
                return -1;
            default:
                return queued_now ? 2 : 1; // queued or encoding
        }
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // [TAG_BS_LAZY_GRAMMAR] n_cmpl > 1: every child takes its first token from the parent's prompt row
            // (copy_state_to shares i_batch), and a backend-sampled token in that row was drawn by the PARENT's chain,
            // so all completions started with the same token instead of independent draws. Tool requests used to be
            // immune because any grammar turned backend sampling off. Sample parents and children on the CPU, where
            // each draws from the shared raw logits with its own chain (found in review).
            use_backend_sampling &= !(task.is_parent() || task.is_child());

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }
            slot.bs_attached = use_backend_sampling;   // [TAG_BS_LAZY_GRAMMAR]

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.t_run_us = ggml_time_us(); // [TAG_POOL_PREEMPT]

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    // [TAG_SPEC_CKPT_PROBE] cumulative accounting for the per-step speculative
    // checkpoint work. 0 = update_tgt, 1 = load_dft, 2 = seq_rm(dft).
    static void spec_ckpt_probe_add(int which, double ms, size_t bytes) {
        static const bool on = [] {
            const char * e = getenv("SPEC_CKPT_PROBE");
            return e && e[0] == '1';
        }();
        if (!on) { return; }
        static double acc[3] = {0,0,0};
        static unsigned n[3] = {0,0,0};
        static size_t last_bytes = 0;
        acc[which] += ms; n[which]++;
        if (bytes) { last_bytes = bytes; }
        if (which == 0 && (n[0] & 127u) == 0u) {
            fprintf(stderr,
                "spec-ckpt-probe: update_tgt n=%u %.3f ms/call (%.2f MiB) | load_dft n=%u %.3f | seq_rm n=%u %.3f\n",
                n[0], acc[0]/n[0], (double) last_bytes / 1024.0 / 1024.0,
                n[1], n[1] ? acc[1]/n[1] : 0.0, n[2], n[2] ? acc[2]/n[2] : 0.0);
            fflush(stderr);
        }
    }

    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // [TAG_SYNC_CKPT_EVICT] Upstream 5d806aa25 (#28302) applies the min-step spacing eviction only once
        // the list is full, counted against --ctx-checkpoints, and supersedes a checkpoint at the same
        // n_tokens instead of appending a duplicate. Here "full" is the effective cap after the byte budget
        // below ([TAG_CKPT_BYTE_BUDGET], 13 entries instead of 32 on Qwen3.8-27B), otherwise the budget's
        // FIFO eviction would always run first and the spacing rule would never fire. When the budget does
        // not bind, the cap is --ctx-checkpoints and this is upstream's rule exactly. The supersede runs
        // first so that a duplicate never costs an unrelated old checkpoint, and it donates its buffer to
        // [TAG_CKPT_BUFFER_REUSE]. LLAMA_CTX_CHECKPOINT_MIN_STEP_ALWAYS=1 restores the pre-sync fork
        // behaviour for A/B: spacing eviction on every call, before the budget, and no supersede.
        static const bool min_step_always = [] {
            const char * e = getenv("LLAMA_CTX_CHECKPOINT_MIN_STEP_ALWAYS");
            return e != nullptr && atoi(e) != 0;
        }();

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        if (!min_step_always) {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    if (slot.ckpt_spare.empty()) {
                        slot.ckpt_spare = std::move(it->data_tgt);
                    }
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        const auto evict_min_step = [&]() {
            int64_t last = -1;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step) {
                    SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                            it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                    it = slot.prompt.checkpoints.erase(it);
                    continue;
                }

                last = it->n_tokens;
                ++it;
            }
        };

        if (min_step_always) {
            evict_min_step();
        }

        // [TAG_CKPT_BYTE_BUDGET]
        // Bound the ring by BYTES as well as by count.
        //
        // A checkpoint is a snapshot of the sequence's recurrent state, taken with
        // LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY. On an attention-only model that flag skips
        // almost everything and an entry costs a few MiB, which is what makes a default of
        // 32 entries reasonable. On a hybrid model it is the opposite: llama_memory_hybrid
        // skips the attention KV and writes the WHOLE recurrent state, which is flat in
        // context and does not shrink with prompt length. For Qwen3.8-27B that is 149.6 MiB
        // per entry no matter how short the prompt, so the same 32-entry default silently
        // reserves 4.7 GiB of host RAM per slot. Measured on a 5090 box: the ring plus the
        // host prompt cache accounted for the whole of an apparent "memory leak".
        //
        // A count is therefore the wrong unit for this bound: it means a hundred times more
        // memory on one model than another. Convert it to a byte budget using the size of
        // an entry we have actually created, so the cap self-tunes per model. This can only
        // ever LOWER the effective count, never raise it above --ctx-checkpoints.
        //
        // Set LLAMA_CTX_CHECKPOINT_BUDGET_MIB=0 to restore the pure count-based behaviour,
        // or to any other number of MiB to choose a different budget.
        size_t n_ckpt_max = (size_t) params_base.n_ctx_checkpoints;

        if (!slot.prompt.checkpoints.empty()) {
            static const size_t budget_bytes = [] {
                const char * e = getenv("LLAMA_CTX_CHECKPOINT_BUDGET_MIB");
                // Default 2048 MiB. On this model that is 13 checkpoints instead of 32,
                // which is still twice the depth reached in real coding sessions.
                const size_t mib = e ? (size_t) strtoull(e, nullptr, 10) : 2048;
                return mib * 1024 * 1024;
            }();

            const size_t one = slot.prompt.checkpoints.back().size();

            if (budget_bytes > 0 && one > 0) {
                // At least one, otherwise a model with a single oversized entry could never
                // checkpoint at all and would re-prefill from scratch every time.
                const size_t by_bytes = std::max<size_t>(1, budget_bytes / one);

                if (by_bytes < n_ckpt_max) {
                    if (!slot.logged_ckpt_budget) {
                        slot.logged_ckpt_budget = true;

                        SLT_WRN(slot, "context checkpoints capped at %d by the byte budget instead of %d by count "
                                "(%.1f MiB each, budget %zu MiB) - set LLAMA_CTX_CHECKPOINT_BUDGET_MIB to change\n",
                                (int) by_bytes, params_base.n_ctx_checkpoints,
                                (float) one / 1024 / 1024, budget_bytes / 1024 / 1024);
                    }

                    n_ckpt_max = by_bytes;
                }
            }
        }

        // [TAG_SYNC_CKPT_EVICT] only when the list is full, otherwise short prompts keep just the oldest checkpoint
        if (!min_step_always && slot.prompt.checkpoints.size() + 1 >= n_ckpt_max) {
            evict_min_step();
        }

        while (slot.prompt.checkpoints.size() >= n_ckpt_max) {
            // make room for the new checkpoint, if needed
            auto & old = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    old.pos_min, old.pos_max, old.n_tokens, (float) old.size() / 1024 / 1024);

            // [TAG_CKPT_BUFFER_REUSE] Keep the evicted entry's buffer for the checkpoint about to be
            // created. update_tgt() then resizes to the same size, a no-op, instead of value-initialising
            // and page-faulting a fresh 149.6 MiB allocation on every checkpoint.
            if (slot.ckpt_spare.empty()) {
                slot.ckpt_spare = std::move(old.data_tgt);
            }

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        if (!slot.ckpt_spare.empty()) {
            cur.data_tgt    = std::move(slot.ckpt_spare);
            slot.ckpt_spare = std::vector<uint8_t>();
        }

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);
        slot.ckpt_build_iter = pool_iter; // [TAG_POOL_PREEMPT] dropped again if its rows are never decoded

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    // [TAG_SHARED_PREFIX_FANOUT] wait for a slot that is already processing this
                    // same prefix rather than processing all of it again on another slot
                    if (share_prefix_should_defer(task)) {
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        // [TAG_BS_LAZY_GRAMMAR] This is the one place the reasoning budget enters FORCING outside
                        // common_sampler_accept, where the detach check normally runs. The web UI arms
                        // reasoning_control on every request, so its slots now keep the backend sampler; left attached,
                        // the next decode would backend-sample rows the forced end sequence must overwrite, and
                        // common_sampler_sample's exactness assert would abort the whole server (found in review).
                        // CONTROL is declined while a decode is in flight, so there are no pending backend rows here.
                        if (slot->bs_attached && !common_sampler_backend_ok(slot->smpl.get())) {
                            common_sampler_backend_detach(slot->smpl.get(), slot->ctx_tgt, slot->id);
                            slot->bs_attached = false;
                        }
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;
                    int n_media_waiting    = 0; // [TAG_MTMD_ASYNC_ENCODE]

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                        if (slot.media_blocked()) {
                            n_media_waiting++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->n_media_waiting     = n_media_waiting;
                    res->n_media_jobs        = media_async ? media_encoder.n_jobs() : 0;
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    // [TAG_ITERATE_WHERE] The catch-all below reports only the exception message, which is
    // identical for every std::vector::at() in the process. That is not enough to localise a
    // 1-in-50 mid-generation failure across eight call sites, so each site names itself and the
    // slot's state is printed alongside. Diagnostics only.
    static void iterate_report(const server_slot & slot, const char * where, const std::exception & e,
                               std::string & msg) {
        msg = std::string("got exception in ") + where + ": " + e.what();
        SLT_ERR(slot, "%s | state=%d n_prompt=%d n_gen=%d n_ctx=%d id_task=%d\n",
                msg.c_str(), (int) slot.state, (int) slot.prompt.n_tokens(),
                (int) slot.stats.n_gen, (int) slot.n_ctx,
                slot.task ? slot.task->id : -1);
    }

    void iterate(const char * where, std::vector<server_slot> & slots,
                 std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                std::string msg;
                iterate_report(slot, where, e, msg);
                send_error(slot, msg, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(const char * where, std::vector<server_slot *> & slots,
                 std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                std::string msg;
                iterate_report(*slot, where, e, msg);
                send_error(*slot, msg, ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                // [TAG_MTMD_ASYNC_ENCODE] The encode of a request that is gone (cancelled, disconnected) may still be
                // running, a clip encode cannot be interrupted. Keep the loop turning until it ends, as a waiting slot
                // would: otherwise the idle timer could put the server to sleep, and destroy() would have to wait for
                // the encode while holding the task queue. The wait returns when the encoder is idle, or after 5 ms.
                if (media_async && media_encoder.n_jobs() > 0) {
                    metrics_flush_idle();

                    server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                    task.id = queue_tasks.get_new_id();
                    queue_tasks.post(std::move(task));

                    media_encoder.wait_idle(std::chrono::milliseconds(5));
                    return;
                }

                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);

                    if (pool_preempt) {
                        pool_no_progress = 0;
                        if (off >= batch.n_gen_rows) {
                            pool_dbg_prompt_views++;
                        }
                    }
                } else {
                    // [TAG_POOL_PREEMPT] the pool cannot take even the smallest valid view: undo the tail, make room, rebuild
                    if (pool_exhausted) {
                        pool_exhausted = false;
                        try {
                            pool_handle_exhausted(off, nullptr);
                        } catch (const std::exception & e) {
                            SRV_ERR("pool_handle_exhausted() failed: %s\n", e.what());
                            abort_all_slots("pool preemption failed: " + std::string(e.what()));
                        }
                        break;
                    }

                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        // [TAG_POOL_PREEMPT] an image that does not fit was held back by the prompt pass: make room for it now
        if (pool_want_room >= 0) {
            const int id_wait = pool_want_room;
            pool_want_room = -1;
            if (!pool_handled_iter && id_wait < (int) slots.size()) {
                try {
                    pool_handle_exhausted(batch.size(), &slots[id_wait]);
                } catch (const std::exception & e) {
                    SRV_ERR("pool_handle_exhausted() failed: %s\n", e.what());
                    abort_all_slots("pool preemption failed: " + std::string(e.what()));
                }
            }
        }
        if (pool_preempt && batch.size() == 0 && pool_n_parked > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // parked slots waiting, nothing to decode: do not spin a core
        }

        // [TAG_MTMD_ASYNC_ENCODE] Nothing was decoded because the only work left is slots waiting for their media
        // encode. update_slots() posted NEXT_RESPONSE above, so the loop comes straight back here: instead of
        // spinning a core for minutes, wait until one of those encodes finishes (the encoder thread notifies), or
        // 5 ms at most so that new tasks are still picked up promptly. The loop keeps turning, so the idle timer
        // cannot put the server to sleep (and free mctx) under a running encode.
        if (media_async && batch.size() == 0) {
            std::vector<const server_media_job *> waiting;
            for (const auto & slot : slots) {
                if (slot.media_blocked()) {
                    waiting.push_back(slot.media_job.get());
                }
            }
            if (!waiting.empty()) {
                media_encoder.wait_any(waiting, std::chrono::milliseconds(5));
            }
        }

        // [TAG_SHARED_PREFIX_FANOUT] the state is only valid once the batch has actually been
        // decoded, so the boundary publish happens here rather than while the batch is built
        share_prefix_publish_ready();
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate("slots_3219", slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && !slot.parked && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // [TAG_POOL_PREEMPT] bring parked slots back before the batch is built
        pool_iter++;
        pool_handled_iter = false;
        if (pool_preempt && pool_n_parked > 0) {
            pool_unpark();
        }

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate("slots_3291", slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || slot.parked) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        // [TAG_SPEC_PROMPT_LAZY] get_text_tokens() reserves and refills a fresh
                        // vector of the WHOLE prompt every speculative step: at 254K that is a
                        // ~1 MB allocation plus 254k branchy push_backs, and the old buffer is
                        // freed - on Windows a 1 MB block round-trips through VirtualAlloc, so
                        // it re-faults every step. Only the n-gram drafters ever read it; DFlash2
                        // and MTP never touch dparams.prompt, so skip the copy entirely for them.
                        if (common_speculative_wants_prompt(spec.get())) {
                            slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
                        } else if (!slot.spec_prompt.empty()) {
                            slot.spec_prompt.clear();
                        }

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            // [TAG_SPEC_MEDIA_POS] pos_next(), NOT n_tokens(). Every drafter uses
                            // pos0 as a POSITION, but n_tokens() is tokens.size(), which counts a
                            // media chunk as its full row count. For a 20x13 image that is 260
                            // against a true advance of 20, so draft() would be asked to draft 240
                            // positions past where the drafter's cache actually ends. pos_next()
                            // applies the n_pos - n_tokens discount per media chunk.
                            // No-op for text: with no media chunks pos_next() == tokens.size().
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .dists    = */ &slot.spec_dists,
                            /* .temperature = */ slot.task->params.sampling.temp,
                            /* .seed     = */ common_sampler_get_seed(slot.smpl.get()),
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate("drafting_3378", drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    const auto ck_d0 = std::chrono::steady_clock::now();
                    ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    spec_ckpt_probe_add(1, std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ck_d0).count(), 0);
                }

                const auto ck_r0 = std::chrono::steady_clock::now();
                const bool ck_rm_ok = llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1);
                spec_ckpt_probe_add(2, std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - ck_r0).count(), 0);
                if (!ck_rm_ok) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    // [TAG_SPEC_CKPT_PROBE] This saves the target sequence state EVERY
                    // speculative step when the draft is longer than the recurrent-state
                    // rollback capacity. Qwen3.8-27B is 48 Gated DeltaNet layers, whose
                    // state cannot be partially rolled back, so the cost scales with the
                    // context. SPEC_CKPT_PROBE=1 reports it.
                    const auto ck_t0 = std::chrono::steady_clock::now();
                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    spec_ckpt_probe_add(0, std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - ck_t0).count(), ckpt.size());

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate("generating_3441", generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });
        batch.n_gen_rows = batch.size(); // [TAG_POOL_PREEMPT] sampled + draft rows lead the batch

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate("slots_3456", slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing() || slot.parked) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // [TAG_MTMD_ASYNC_ENCODE] The prompt is at a media chunk that is still being encoded on the
                    // encoder thread: add nothing for this slot, the other slots are batched as usual. Checked
                    // before anything below touches the batch, and before print_timings_pp(), which would log on
                    // every iteration of the wait. This also queues the encode of the next media chunk while the
                    // prompt is still on text, so that the encode overlaps with that text.
                    if (media_async) {
                        const int r = media_async_step(slot);
                        if (r < 0) {
                            SLT_ERR(slot, "%s", "failed to process mtmd chunk, the encode failed\n");
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }
                        if (r > 0) {
                            // The wait starts here (r == 2, the job was just queued): drop what lies past the
                            // prompt from memory now, as the synchronous path does before its encode (the seq_rm
                            // below), so that a waiting slot holds exactly its prompt's cells - the pool, a park
                            // and a restore see the same state. Only once, it is a scan over the whole KV cache.
                            if (r == 2) {
                                slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
                            }
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // [TAG_MTMD_ASYNC_ENCODE] the embeddings come from the slot's encode job. A second media chunk
                        // right after the first may still be encoding, then the slot waits from here: the chunks
                        // decoded so far stay decoded, and the text after them is added once the rest is decoded,
                        // so has_mtmd (no checkpoint in that batch) comes out as in the synchronous path.
                        float * embd_async = nullptr;
                        if (media_async) {
                            const int r = media_async_step(slot);
                            if (r > 0) {
                                return;
                            }
                            embd_async = (r == 0 && slot.media_job) ? slot.media_job->get_embd(cur_token_idx) : nullptr;
                            if (embd_async == nullptr) {
                                SLT_ERR(slot, "failed to process mtmd chunk, no embeddings for idx = %zu\n", (size_t) cur_token_idx);
                                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }
                        }

                        // [TAG_POOL_PREEMPT] an image decodes in its own llama_decode calls, and a full pool used to fail the request.
                        // Hold it back instead (no rows added this iteration) and let update_slots make room.
                        if (pool_preempt) {
                            const auto & chunk_pool = input_tokens.find_chunk(cur_token_idx);
                            const int64_t n_rows_pool = (int64_t) mtmd_input_chunk_get_n_tokens(chunk_pool.get());
                            if ((int64_t) llama_memory_attn_n_free_ext(ctx_tgt, slot.id) < n_rows_pool) {
                                SLT_WRN(slot, "__TEST_TAG_POOL_MEDIA_WAIT__ image needs %lld cells\n", (long long) n_rows_pool);
                                pool_want_room = slot.id;
                                return;
                            }
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            // [TAG_MTMD_ENCODE_CATCH] mtmd_batch_encode already turns a std::exception from
                            // the encoder (a Vulkan DeviceLost on the iGPU is a vk::SystemError) into an error
                            // code. This also stops anything else thrown here (a non-std exception, the image
                            // decode helper) from escaping update_slots and killing the server with every
                            // slot. Fail only this request: res != 0 below sends the error and releases the
                            // slot, which also drops the partially encoded slot.mbatch.
                            try {
                                // [TAG_MTMD_ASYNC_ENCODE] async: only the decode is left, the encode ran on the encoder thread
                                res = embd_async != nullptr
                                    ? process_mtmd_chunk_decode(slot, cur_token_idx, embd_async, n_tokens_out)
                                    : process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                            } catch (const std::exception & e) {
                                SLT_ERR(slot, "exception while processing mtmd chunk: %s\n", e.what());
                                res = -1;
                            } catch (...) {
                                SLT_ERR(slot, "%s", "unknown exception while processing mtmd chunk\n");
                                res = -1;
                            }
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        // [TAG_MTMD_ASYNC_ENCODE] free the job (chunk copies, embeddings) once its last chunk is decoded
                        if (embd_async != nullptr && slot.media_job && slot.media_job->idxs.back() == (size_t) cur_token_idx) {
                            slot.media_job.reset();
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    // [TAG_MTMD_ASYNC_ENCODE] media was just decoded and the prompt goes on with text: queue the
                    // next media chunk now, so that its encode overlaps with this text. The result only matters for
                    // a failure to queue, which is reported when the prompt reaches that chunk.
                    if (media_async && has_mtmd && !slot.media_job) {
                        (void) media_async_step(slot);
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // [TAG_CKPT_TURN_MERGE] see the two break sites below. LLAMA_SERVER_TURN_SPLIT=1 restores the
                    // upstream batching: a separate batch for the newline before the user message and another
                    // for the last 4 tokens, each with its own decode and its own 149.6 MiB checkpoint.
                    static const bool turn_split_forced = [] {
                        const char * e = getenv("LLAMA_SERVER_TURN_SPLIT");
                        return e != nullptr && atoi(e) != 0;
                    }();
                    // a user-start break is skipped while this batch holds at most this many of the slot's tokens
                    constexpr int64_t ckpt_merge_max = 8;
                    // the tail split (last 4 tokens as their own batch and checkpoint) only pays for itself when
                    // the batch already carries at least this many tokens that a regenerate would re-prefill
                    constexpr int64_t ckpt_tail_min = 512;

                    bool user_start_merged         = false;
                    bool user_start_merged_is_last = false;

                    // add prompt tokens for processing in the current batch
                    // [TAG_SHARED_PREFIX_FANOUT] stop exactly on the boundary so the state saved
                    // after this batch is the shared prefix and nothing more
                    const int share_stop = slot.share_prefix_stop;

                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch
                           && (share_stop == 0 || (int) slot.prompt.n_tokens() < share_stop)) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                // [TAG_CKPT_TURN_MERGE] The break exists so that the next batch starts exactly at the
                                // user message and gets its checkpoint. On every follow-up turn the only tokens before
                                // it are the newline after the previous reply, so the break produced a 1-token batch:
                                // one extra decode plus one extra checkpoint, one token away from the next one. When
                                // this batch holds only a few tokens so far, keep filling and let the checkpoint at
                                // this batch's start stand for the user message. An edit of that message then
                                // re-prefills those few tokens more, nothing else changes.
                                const int64_t n_in_batch = (int64_t) (batch.size() - n_tokens_prev);
                                if (turn_split_forced || user_start_merged || n_in_batch > ckpt_merge_max) {
                                    break;
                                }
                                user_start_merged         = true;
                                user_start_merged_is_last = pos == last_user_pos;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            // [TAG_CKPT_TURN_MERGE] the tail split only pays for itself past ckpt_tail_min tokens
                            if (should_break && (turn_split_forced || (int64_t) (batch.size() - n_tokens_prev) >= ckpt_tail_min)) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start) || user_start_merged;
                    const bool is_last_user_message = n_tokens_start == last_user_pos || user_start_merged_is_last;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
        }

        // [TAG_SPEC_PREFILL_TAIL_EXTRACT] Publish how many prompt tokens follow this batch BEFORE the
        // target decode (this used to happen after it, right before the drafter ran). With that known,
        // ask the drafter whether it will skip this batch, and if so let the target skip extracting the
        // five per-layer inputs the drafter would never read: 5 x n_ubatch x n_embd x 4 B, 131 MB per
        // 1280-token ubatch, copied GPU to host on the compute stream, 13 GB over a 131k prompt.
        // SPEC_PREFILL_EXTRACT_ALL=1 restores the unconditional extraction.
        bool skip_extract = false;
        if (spec) {
            // [TAG_SPEC_PREFILL_TAIL_PER_SEQ] Publish BOTH: the max, which the is_prefill
            // heuristic wants, and the per-slot value, which the skip decision needs. The
            // skip wipes the drafter KV of every sequence in the batch, so letting the max
            // speak for all of them meant a generating slot co-batched with a prefilling one
            // lost its drafter cache on every batch, collapsing its acceptance to ~0.
            // The sequence id IS the slot id, see common_batch_add(..., { t.id_slot }, ...).
            common_speculative_clear_prefill_after_seq(spec.get());

            int32_t n_after = 0;
            for (const auto & slot : slots) {
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT && slot.task && !slot.parked) {
                    const int32_t total = (int32_t) slot.task->n_tokens();
                    const int32_t done  = (int32_t) slot.prompt.n_tokens();
                    const int32_t rem   = total - done;
                    n_after = std::max(n_after, rem);
                    // [TAG_SHARE_PREFIX_DFT_TAIL] While a slot is paused on a shared boundary
                    // ([TAG_SHARED_PREFIX_FANOUT]) the drafter state that gets published is the one AT
                    // that boundary, so its warm tail has to end there. Measured to the full prompt end
                    // instead, a leader whose own tail runs 2048+ tokens past the boundary skipped (and
                    // wiped) the drafter for every batch up to it, and each follower restored an EMPTY
                    // drafter window and drafted blind until it had generated its way back. The scalar
                    // max above still measures to the full end, which is what the is_prefill heuristic
                    // wants. TURBO_SHARE_PREFIX_DFT_TAIL=0 restores the old distance.
                    static const bool dft_tail_to_boundary = [] {
                        const char * e = getenv("TURBO_SHARE_PREFIX_DFT_TAIL");
                        return e == nullptr || atoi(e) != 0;
                    }();
                    const int32_t skip_end = (dft_tail_to_boundary && slot.share_prefix_stop > 0)
                                               ? std::min<int32_t>(total, (int32_t) slot.share_prefix_stop)
                                               : total;
                    common_speculative_set_prefill_after_seq(spec.get(), slot.id, std::max<int32_t>(0, skip_end - done));
                }
            }
            common_speculative_set_prefill_after(spec.get(), n_after);

            static const bool extract_all = [] {
                const char * e = getenv("SPEC_PREFILL_EXTRACT_ALL");
                return e != nullptr && atoi(e) != 0;
            }();
            skip_extract = !extract_all && common_speculative_prefill_will_skip(spec.get(), batch_view);
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        // [TAG_SPEC_PHASE_PROBE]
        const auto t_dec0 = common_speculative_probe_enabled()
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
        queue_tasks.yield_to_queue([&]() {
            if (skip_extract) {
                llama_set_layer_inp_extract(ctx_tgt, false);
            }
            ret = pool_dbg_fire(off) ? 1 : llama_decode(ctx_tgt, batch_view); // [TAG_POOL_PREEMPT] hook inert unless TURBO_POOL_DEBUG_FAIL
            if (skip_extract) {
                llama_set_layer_inp_extract(ctx_tgt, true);
            }
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        // Count only generation-sized batches. A prefill ubatch (512) also lands here
        // and would otherwise be averaged into the per-step figure.
        if (common_speculative_probe_enabled() && batch_view.n_tokens <= 64) {
            common_speculative_probe_add(COMMON_SPEC_PHASE_TGT_DECODE,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec0).count());
        }

        if (ret != 0) {
            // [TAG_POOL_PREEMPT] never halve through a verify group and never fail every slot
            if (pool_preempt && ret == 1) {
                GGML_ASSERT(off == 0 || off >= batch.n_gen_rows);
                if (pool_dbg_firing < 0 && try_clear_idle_slots(off)) {
                    SRV_WRN("[TAG_POOL_PREEMPT] KV pool full at off = %d, cleared an idle slot, retrying\n", off);
                    return false;
                }
                const int32_t floor  = off < batch.n_gen_rows ? batch.n_gen_rows - off : 1;
                const int32_t n_free = llama_memory_attn_n_free_ext(ctx_tgt, 0);
                if (n_free < floor || batch_view.n_tokens <= floor) {
                    pool_exhausted = true;
                    return false;
                }
                n_batch = std::max(floor, std::min(batch_view.n_tokens / 2, n_free));
                SRV_WRN("[TAG_POOL_PREEMPT] KV pool full at off = %d, n_free = %d, retry with n_batch = %d\n", off, n_free, n_batch);
                return false;
            }

            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            // [TAG_SPEC_PREFILL_TAIL]
            // Tell the drafter how many prompt tokens still follow this ubatch. A drafter
            // whose attention is a sliding window (DFlash2: 2048) only needs a warm KV for
            // the tail, so it can skip the earlier ones - which is otherwise a full encode
            // and decode of the draft model per prompt ubatch.
            //
            // Reported as the max over the slots still processing their prompt; 0 once
            // every slot is generating, so generation is never skipped.
            {
                // [TAG_SPEC_PREFILL_TAIL_PER_SEQ] Publish BOTH: the max, which the is_prefill
                // heuristic wants, and the per-slot value, which the skip decision needs. The
                // skip wipes the drafter KV of every sequence in the batch, so letting the max
                // speak for all of them meant a generating slot co-batched with a prefilling one
                // lost its drafter cache on every batch, collapsing its acceptance to ~0.
                // The sequence id IS the slot id, see common_batch_add(..., { t.id_slot }, ...).
                // [TAG_SPEC_PREFILL_TAIL_EXTRACT] the values for this batch were published before the
                // target decode above, nothing between there and here changes them
            }

            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()
                    && slot.i_batch >= off && slot.i_batch < off + batch_view.n_tokens) { // [TAG_POOL_PREEMPT] only once the parent's last row is decoded
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate("slots_4238", slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate("slots_4251", slots, [&](server_slot & slot) {
            if (slot.parked) {
                return; // [TAG_POOL_PREEMPT]
            }

            // optionally send prompt processing progress
            // [TAG_MTMD_ASYNC_ENCODE] not while the prompt waits for its media encode: nothing changed, and the other
            // slots' decodes would send the same progress event every step for the whole encode
            if ((slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) && !slot.media_blocked()) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // [TAG_BS_LAZY_GRAMMAR] the lazy grammar just triggered (or the reasoning budget started forcing):
            // this slot samples on the CPU, grammar applied, from the next decode on
            if (slot.bs_attached && !common_sampler_backend_ok(slot.smpl.get())) {
                common_sampler_backend_detach(slot.smpl.get(), slot.ctx_tgt, slot.id);
                slot.bs_attached = false;
            }

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate("slots_4344", slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

// [TAG_PHASE_PROBE] name the phase that throws, so the next occurrence localises itself
#define TURBO_PHASE_BEGIN try {
#define TURBO_PHASE_END(name) } catch (const std::exception & e_) { \
    throw std::runtime_error(std::string(name) + ": " + e_.what()); }

            // verify and try to accept the draft
            TURBO_PHASE_BEGIN
            {
                // [TAG_SPEC_SMPL_CLONE] Only the use_ckpt_tgt branch below consumes this clone, and
                // cloning is not cheap: common_sampler_clone copies gsmpl->cur, which is
                // n_vocab * sizeof(llama_token_data) = 248320 * 12 = 2.98 MB, plus a deep copy of the
                // grammar stacks and the rest of the chain. That ran on EVERY speculative step.
                //
                // Hoist the same predicate the consumer uses. With qwen35 (which is in
                // llm_arch_supports_rs_rollback) the type is RS and n_rollback <= n_draft <= n_max ==
                // llama_n_rs_seq, so use_ckpt_tgt is always false here and the clone was pure waste.
                const bool may_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
                     // [TAG_SPEC_CKPT_OFF_BY_ONE] n_draft, NOT n_draft + 1. The consumer below
                     // tests n_rollback > n_rs_seq, and n_rollback = n_draft + 1 - accepted.size()
                     // with accepted.size() >= 1, so n_rollback <= n_draft. Hoisting n_draft + 1
                     // was one too many and made this true on every full-width step at the shipped
                     // default (n_rs_seq == n_max == 7 gives 8 > 7), which is precisely the case the
                     // comment above says should never clone.
                     (uint32_t) n_draft > llama_n_rs_seq(ctx_tgt));
                
                common_sampler_ptr smpl_save(may_ckpt_tgt ? common_sampler_clone(slot.smpl.get()) : nullptr);

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                // Two independent additions land on the same dispatch and both must survive:
                //   - upstream's synthesised-probability drafting (server_sample_and_accept_synth)
                //   - our rollback-aware path, which needs the per-draft distributions so a
                //     rejected draft can be partially rolled back instead of replayed.
                // Synth is a property of the drafter, so it wins when present; otherwise fall
                // back to our rollback path, and to the plain accept when rollback is not
                // available (no PART/RS seq-rm support, greedy sampling, or a dists/draft
                // length mismatch).
                const bool can_rollback =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft <= llama_n_rs_seq(ctx_tgt));
                const auto t_smp0 = common_speculative_probe_enabled()
                                      ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted =
                    !synth_probs.empty()
                        ? server_sample_and_accept_synth(
                                slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                                synth_probs, slot.spec_synth_rng, slot.spec_is_replay)
                    : (can_rollback && slot.task->params.sampling.temp > 0.0f &&
                       slot.spec_dists.size() == slot.spec_draft.size())
                        ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft, slot.spec_dists)
                        : common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft);
                if (common_speculative_probe_enabled()) {
                    common_speculative_probe_add(COMMON_SPEC_PHASE_SAMPLE,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_smp0).count());
                    common_speculative_probe_step();
                }
                slot.spec_i_batch.clear();

                // [TAG_BS_LAZY_GRAMMAR] accept stopped at the token that activated the grammar or the budget's forcing;
                // switch this slot to CPU sampling before its next verify decode is built
                if (slot.bs_attached && !common_sampler_backend_ok(slot.smpl.get())) {
                    common_sampler_backend_detach(slot.smpl.get(), slot.ctx_tgt, slot.id);
                    slot.bs_attached = false;
                }

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);
                        slot.spec_dists.clear();

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        // may_ckpt_tgt upper-bounds n_rollback by n_draft+1, so it can never
                        // be false while use_ckpt_tgt is true. Assert rather than deref null.
                        GGML_ASSERT(smpl_save && "sampler clone predicate diverged from its use");
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
                slot.spec_dists.clear();
            }
            TURBO_PHASE_END("accept")

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;

            TURBO_PHASE_BEGIN
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
            TURBO_PHASE_END("bookkeep")

            TURBO_PHASE_BEGIN
            // [TAG_BAD_TOKEN_GUARD] The vocab indexes std::vectors by token id with .at()
            // (llama-vocab.cpp:3150, :3698), so an id outside [0, n_vocab) throws
            // "invalid vector subscript" and the whole request dies with an HTTP 500. That is
            // what has been happening on roughly 1 in 40 long reasoning generations, in stock
            // llama.cpp as well as here. Report the actual value and stop cleanly instead.
            const int n_vocab_tgt =
                llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(slot.ctx_tgt)));

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok = ids[i];

                if (result.tok < 0 || result.tok >= n_vocab_tgt) {
                    SLT_ERR(slot, "[TAG_BAD_TOKEN_GUARD] speculative accept produced token id %d, "
                            "outside [0, %d) - position %zu of %zu accepted, n_draft = %zu, "
                            "replay = %d. Dropping it and ending this generation cleanly rather "
                            "than throwing.\n",
                            (int) result.tok, n_vocab_tgt, i, ids.size(), n_draft,
                            (int) slot.spec_is_replay);

                    slot.stop           = STOP_TYPE_EOS;
                    slot.has_next_token = false;

                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }

                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();
            TURBO_PHASE_END("emit")

#undef TURBO_PHASE_BEGIN
#undef TURBO_PHASE_END

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot
    // [TAG_SYNC_NCTX_NOCAP] upstream also caps to the training context of the model; we deliberately do not
    // (YaRN / linear rope scaling wants the longer window), load_model() warns when no scaling is configured
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return res;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            // [TAG_TURN_TIMING] the whole conversation is tokenized on every turn; make its cost visible
            const int64_t t_tok0 = ggml_time_us();
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
            SRV_INF("tokenized prompt: %.1f ms, %zu tokens\n", (ggml_time_us() - t_tok0) / 1000.0,
                    inputs.empty() ? (size_t) 0 : inputs[0].size());
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
