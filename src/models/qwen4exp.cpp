#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"
#include "llama-moetrace.h" // [TAG_FN_MOE_TRACE]
#include "llama-moe-gen5.h" // [TAG_MOE_PREFETCH]
#include "llama-ple-dio.h"  // [TAG_FN_PLE_DIRECT_IO]
#include "llama-ext.h"      // [TAG_FN_SHIP1] llama_model_fn_env

#include "ggml-alloc.h"   // [TAG_FN_MTP_HEAD_IDS]
#include "ggml-backend.h"
#include "ggml-fn-l3.h"   // [TAG_FN_L3_GPU]

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>     // [TAG_FN_SHIP1]
#include <fstream>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// [TAG_FN_PLE_DIRECT_IO] LLAMA_PLE_DIRECT_IO=1: open the unbuffered row reader for the PLE table, or nullptr (the table
// then stays mapped, as without the switch). The loader was asked (TENSOR_READ_DIRECT) to keep the table lazy in any
// load mode and never to read, validate, prefetch or lock it.
static void qwen4exp_dio_log(const char * line) {
    if (strstr(line, "warning:") != nullptr) {
        LLAMA_LOG_WARN("%s", line);
    } else {
        LLAMA_LOG_INFO("%s", line);
    }
}

// 512 rows (256 spread over the table with the first and last, 256 scattered) read from both files must match.
// Both readers are unbuffered, with no LRU and no injected failures; the mapping is not touched.
static bool qwen4exp_ple_copy_matches(const std::string & copy_path, const llama_ple_dio_params & shard) {
    llama_ple_dio_params q = shard;
    q.cache_bytes     = 0;
    q.stats_every     = 0;
    q.test_fail_every = 0;
    std::string err;
    auto ref = llama_ple_dio::open(q, err);
    if (!ref) {
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] cannot read the model file to check the copy: %s\n", __func__, err.c_str());
        return false;
    }
    q.path   = copy_path;
    q.offset = 0;
    auto copy = llama_ple_dio::open(q, err);
    if (!copy) {
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] cannot read %s: %s\n", __func__, copy_path.c_str(), err.c_str());
        return false;
    }
    const int n = 512;
    std::vector<int32_t> rows(n);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; ++i) {
        if (i < n/2) {
            rows[i] = (int32_t) ((shard.n_rows - 1) * i / (n/2 - 1));
        } else {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            rows[i] = (int32_t) (x % (uint64_t) shard.n_rows);
        }
    }
    std::vector<uint8_t> a(n * shard.row_bytes);
    std::vector<uint8_t> b(n * shard.row_bytes);
    try {
        copy->read_rows(rows.data(), n, a.data(), 0, nullptr);
        ref->read_rows(rows.data(), n, b.data(), 0, nullptr);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] check read failed: %s\n", __func__, e.what());
        return false;
    }
    return a == b;
}

// LLAMA_PLE_DIO_FILE: the table's rows in a file of their own, made here when missing or wrong. nullptr on failure.
static std::unique_ptr<llama_ple_dio> qwen4exp_open_ple_copy(const llama_ple_dio_params & shard, const std::string & path) {
    const uint64_t bytes = (uint64_t) shard.n_rows * shard.row_bytes;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const int64_t have = llama_ple_dio_file_size(path);
        if (attempt > 0 || have != (int64_t) bytes) {
            if (have >= 0 && have != (int64_t) bytes) {
                LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] %s (%lld bytes) is not a copy of this table (%llu bytes): making it again\n",
                        __func__, path.c_str(), (long long) have, (unsigned long long) bytes);
            }
            LLAMA_LOG_INFO("%s: [TAG_FN_PLE_DIRECT_IO] copying the PLE table (%.2f GiB) to %s with unbuffered I/O (once)\n",
                    __func__, bytes / 1073741824.0, path.c_str());
            std::string err;
            if (!llama_ple_dio_copy(shard.path, shard.offset, bytes, path, (size_t) 8 << 20, err, qwen4exp_dio_log)) {
                LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] copy failed: %s\n", __func__, err.c_str());
                return nullptr;
            }
        }
        if (qwen4exp_ple_copy_matches(path, shard)) {
            llama_ple_dio_params q = shard;
            q.path   = path;
            q.offset = 0;
            std::string err;
            auto dio = llama_ple_dio::open(q, err);
            if (!dio) {
                LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] cannot open %s: %s\n", __func__, path.c_str(), err.c_str());
            }
            return dio;
        }
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] %s does not match the table in the model file\n", __func__, path.c_str());
    }
    return nullptr;
}

// [TAG_FN_SHIP1] room for the profile's automatic copy: its size plus 16 GiB must stay free on the copy's volume (a copy
// of the right size that exists needs none), so a default never fills a disk
static bool qwen4exp_ple_auto_room(const llama_ple_dio_params & shard, const std::string & path, std::string & why) {
    const uint64_t bytes = (uint64_t) shard.n_rows * shard.row_bytes;
    if (llama_ple_dio_file_size(path) == (int64_t) bytes) {
        return true;
    }
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::u8path(path).parent_path();
    if (dir.empty()) {
        dir = std::filesystem::current_path(ec);
    }
    const std::filesystem::space_info sp = std::filesystem::space(dir, ec);
    if (ec) {
        why = "the free space of its volume is unknown (" + ec.message() + ")";
        return false;
    }
    const uint64_t need = bytes + (16ull << 30);
    if (sp.available < need) {
        why = format("its volume has %.1f GiB free, the copy needs %.1f GiB plus 16 GiB to spare",
                sp.available / 1073741824.0, bytes / 1073741824.0);
        return false;
    }
    return true;
}

// copy_path: the user's LLAMA_PLE_DIO_FILE, else the qwen4exp profile's <model>.ple (copy_auto), else nullptr
static std::shared_ptr<llama_ple_dio> qwen4exp_open_ple_dio(const llama_model_loader & ml, const ggml_tensor * t,
        const char * copy_path, bool copy_auto) {
    const char * name = ggml_get_name(t);
    auto off = [&](const std::string & why) {
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] LLAMA_PLE_DIRECT_IO is set, but %s; %s is read through the mapping\n",
                __func__, why.c_str(), name);
        return std::shared_ptr<llama_ple_dio>();
    };
    if (!ml.lazy.has(t)) {
        return off("the tensor is not lazily mapped (no mmap support)");
    }
    if (t->type != GGML_TYPE_F32 && ggml_get_type_traits(t->type)->to_float == nullptr) {
        return off(std::string("type ") + ggml_type_name(t->type) + " has no to_float");
    }
    if (t->ne[2] != 1 || t->ne[3] != 1 || (size_t) t->nb[1] != ggml_row_size(t->type, t->ne[0])) {
        return off("the table is not a plain 2-D row array");
    }
    const auto * w = ml.get_weight(name);
    if (w == nullptr) {
        return off("its file position is unknown");
    }
    if (w->idx >= ml.files_paths.size() || ml.files_paths[w->idx].empty()) {
        return off("the model was not loaded from a file path");
    }

    auto env_ll = [](const char * key, long long def) {
        const char * e = getenv(key);
        return e != nullptr && *e != '\0' ? atoll(e) : def;
    };

    llama_ple_dio_params p;
    p.path        = ml.files_paths[w->idx];
    p.offset      = w->offs;
    p.row_bytes   = t->nb[1];
    p.n_rows      = t->ne[1];
    p.cache_bytes = (size_t) std::max(0LL, env_ll("LLAMA_PLE_DIO_CACHE_MB", 64)) << 20;
    p.queue_depth = (int) std::max(1LL, std::min(1024LL, env_ll("LLAMA_PLE_DIO_QD", 64)));
    p.stats_every = (int) std::max(0LL, std::min((long long) INT32_MAX, env_ll("LLAMA_PLE_DIO_STATS", 0)));
    p.log         = qwen4exp_dio_log;
    // tests only: every Nth read fails, so its row comes from the mapped table (must not change any value)
    p.test_fail_every = (int) std::max(0LL, std::min((long long) INT32_MAX, env_ll("LLAMA_PLE_DIO_TEST_FAIL", 0)));

    // LLAMA_PLE_DIO_FILE: read a copy that nothing maps (Windows serves unbuffered reads of a mapped file one at a time)
    // [TAG_FN_SHIP1] the profile's automatic copy (copy_auto) is made only with room to spare on its volume; when it
    // cannot be made or opened, the table is read through the mapping (as without the lever), not from the mapped model
    // file one read at a time
    std::unique_ptr<llama_ple_dio> dio;
    if (copy_path != nullptr && *copy_path != '\0') {
        if (copy_auto) {
            std::string why;
            if (!qwen4exp_ple_auto_room(p, copy_path, why)) {
                return off(std::string("no automatic copy at ") + copy_path + ": " + why +
                           " (LLAMA_PLE_DIO_FILE=<path> puts it elsewhere)");
            }
        }
        dio = qwen4exp_open_ple_copy(p, copy_path);
        if (!dio && copy_auto) {
            return off(std::string("the automatic copy ") + copy_path +
                       " could not be made or opened (LLAMA_PLE_DIO_FILE=<path> puts it elsewhere)");
        }
        if (!dio) {
            LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] LLAMA_PLE_DIO_FILE=%s is not usable; reading the model file instead\n",
                    __func__, copy_path);
        }
    }
    if (!dio) {
        std::string err;
        dio = llama_ple_dio::open(p, err);
        if (!dio) {
            return off("the file cannot be read unbuffered (" + err + ")");
        }
    }
    const auto & dp = dio->params();
    LLAMA_LOG_INFO("%s: [TAG_FN_PLE_DIRECT_IO] %s: rows read %s from %s @ %llu (%lld rows x %zu B, %zu B aligned, "
            "%d in flight, LRU %zu MiB%s%s)\n", __func__, name, dio->direct() ? "unbuffered" : "with pread + DONTNEED",
            dp.path.c_str(), (unsigned long long) dp.offset, (long long) dp.n_rows, dp.row_bytes, dio->align(),
            dp.queue_depth, dp.cache_bytes >> 20, dp.stats_every ? ", stats on" : "",
            dp.test_fail_every ? ", TEST: injected read failures" : "");
#if defined(_WIN32)
    if (dp.path == p.path) {
        LLAMA_LOG_WARN("%s: [TAG_FN_PLE_DIRECT_IO] warning: this file is also mapped, so Windows serves these reads one "
                "at a time (~10K/s, slow prefill); LLAMA_PLE_DIO_FILE=<path> reads an unmapped copy of the table\n", __func__);
    }
#endif
    return std::shared_ptr<llama_ple_dio>(std::move(dio));
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

// [TAG_FN_MTP_HEAD_IDS] token ids in [0, n_vocab) from a text file: separated by spaces, commas or new lines, '#' starts
// a comment (the LLAMA_DFLASH_HEAD_EXTRA format). Empty when the file cannot be read.
static std::vector<int32_t> qwen4exp_read_ids(const char * path, int64_t n_vocab) {
    std::vector<int32_t> ids;
    std::ifstream f(path);
    if (!f) {
        LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_IDS] cannot read '%s'\n", __func__, path);
        return ids;
    }
    std::string line;
    while (std::getline(f, line)) {
        line = line.substr(0, line.find('#'));
        std::replace(line.begin(), line.end(), ',', ' ');
        size_t pos = 0;
        while (pos < line.size()) {
            const size_t beg = line.find_first_not_of(" \t\r", pos);
            if (beg == std::string::npos) {
                break;
            }
            const size_t  end = line.find_first_of(" \t\r", beg);
            const int64_t id  = atoll(line.substr(beg, end - beg).c_str());
            if (id >= 0 && id < n_vocab) {
                ids.push_back((int32_t) id);
            }
            pos = end;
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // QSA pools the indexer keys of blocks of compress_ratio cells, one block size for the whole model
    hparams.indexer_kpool = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t r = hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (hparams.indexer_kpool != 0 && r != hparams.indexer_kpool) {
            throw std::runtime_error(format("QSA layers must share one compress ratio, got %u and %u", hparams.indexer_kpool, r));
        }
        hparams.indexer_kpool = r;
    }
    if (hparams.indexer_kpool == 1 || (hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool != 0)) {
        throw std::runtime_error(format("QSA needs a compress ratio above 1 that divides the budget, got %u and %u",
                                        hparams.indexer_kpool, hparams.indexer_top_k));
    }
    // the reference groups the visible tokens in cache order and always keeps the tail
    hparams.indexer_kpool_row         = 2; // raw key | pooled key
    hparams.indexer_kpool_by_order    = true;
    hparams.indexer_kpool_select_tail = true;

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // an MTP-only file carries the MTP block, the embeddings and the LM head, but no trunk
    const bool mtp_only    = n_layer_nextn > 0 && ml.get_weight(tn(LLM_TENSOR_HC_ATTN_NORM, "weight", 0).str().c_str()) == nullptr;
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int  mtp_flags   = ml.load_mtp ? 0 : TENSOR_SKIP;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        // [TAG_FN_PLE_DIRECT_IO] LLAMA_PLE_DIRECT_IO=1: the table is never mapped in; its rows are read from the file
        const bool ple_direct = llama_ple_dio_requested();
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows },
                                           TENSOR_READ_LAZY | (ple_direct ? llama_model_loader::TENSOR_READ_DIRECT : 0));
        if (ple_direct && per_layer_tok_embd != nullptr && !ml.no_alloc) {
            // [TAG_FN_SHIP1] the user's LLAMA_PLE_DIO_FILE, else the qwen4exp profile's <model>.ple
            const char * user_copy = getenv("LLAMA_PLE_DIO_FILE");
            const char * copy      = llama_model_fn_env(this, "LLAMA_PLE_DIO_FILE");
            const bool   copy_auto = !(user_copy && user_copy[0]) && copy && copy[0];
            ple_dio = qwen4exp_open_ple_dio(ml, per_layer_tok_embd, copy, copy_auto);
        }
    }

    auto load_block = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        load_block(il, trunk_flags);
    }

    // the MTP block: one full-attention QSA layer fed by [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream
    for (int il = n_layer; il < n_layer_all; ++il) {
        load_block(il, mtp_flags);

        auto & nextn = layers[il].nextn;
        nextn.eh_proj      = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,      "weight", il), { 2*n_embd, n_embd }, mtp_flags);
        nextn.enorm        = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,        "weight", il), { n_embd }, mtp_flags);
        // RMS per hc stream of the trunk residual, so the gammas load as [n_embd, hc] like the mixer norms
        nextn.hnorm        = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,        "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, mtp_flags | TENSOR_ALLOW_RESHAPE);
        nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, mtp_flags);
        nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, mtp_flags);
    }

    // [TAG_FN_MTP_HEAD_ROWS] LLAMA_MTP_HEAD_ROWS=N (0 = off, the default): the MTP draft head reads rows [0, N) of the LM
    // head plus the draftable ids above N; the rest of the draft logits is -inf. Only drafts change: verify uses the
    // full head. For the Qwen3.5-family vocabulary 98304 is the value DFlash measured (LLAMA_MTP_HEAD_EXTRA=<file> adds ids).
    if (ml.load_mtp && n_layer_all > n_layer) {
        const char * e = getenv("LLAMA_MTP_HEAD_ROWS");
        const int64_t rows = e ? atoll(e) : 0;
        if (rows > 0 && rows < (int64_t) n_vocab) {
            mtp_head_rows  = rows;
            mtp_head_extra = llama_head_extra_rows(vocab, rows, n_vocab, getenv("LLAMA_MTP_HEAD_EXTRA"));
            LLAMA_LOG_INFO("%s: [TAG_FN_MTP_HEAD_ROWS] MTP drafts read LM head rows [0, %lld) plus %zu more rows (%.1f%% of the head)\n",
                    __func__, (long long) rows, mtp_head_extra.size(), 100.0*(rows + (int64_t) mtp_head_extra.size())/n_vocab);
        }
    }

    // [TAG_FN_MTP_HEAD_IDS] LLAMA_MTP_HEAD_IDS=<file> (unset = off): the MTP drafts score only a calibrated draft
    // vocabulary (e.g. scripts/dflash-draft-vocab.py ids), plus every control and user-defined token. Only drafts
    // change: verify uses the full head. It replaces LLAMA_MTP_HEAD_ROWS when both are set.
    if (ml.load_mtp && n_layer_all > n_layer) {
        const char * path = getenv("LLAMA_MTP_HEAD_IDS");
        if (path && path[0]) {
            std::vector<int32_t> ids = qwen4exp_read_ids(path, n_vocab);
            const size_t n_file = ids.size();
            for (int64_t id = 0; id < (int64_t) n_vocab; ++id) {
                if (vocab.token_get_attr((llama_token) id) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    ids.push_back((int32_t) id);
                }
            }
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            if (n_file == 0 || ids.size() >= (size_t) n_vocab) {
                LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_IDS] '%s' gives %zu ids of %u: MTP drafts use the full head\n",
                        __func__, path, n_file, (uint32_t) n_vocab);
            } else {
                mtp_head_ids = std::move(ids);
                if (mtp_head_rows > 0) {
                    LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_IDS] LLAMA_MTP_HEAD_IDS replaces LLAMA_MTP_HEAD_ROWS\n", __func__);
                    mtp_head_rows = 0;
                    mtp_head_extra.clear();
                }
                LLAMA_LOG_INFO("%s: [TAG_FN_MTP_HEAD_IDS] MTP drafts score %zu ids (%zu from '%s', the rest control tokens), "
                        "%.1f%% of the head\n", __func__, mtp_head_ids.size(), n_file, path, 100.0*mtp_head_ids.size()/n_vocab);
            }
        }
    }

    // [TAG_QWEN4EXP_MTP] [TAG_SYNC_1004] without MTP the block above is skipped, and the generic scale pass of
    // load_tensors keys on the loaded weights, so it never asks for the block's optional ".scale" / ".input_scale"
    // tensors (NVFP4, or any file the saver wrote): skip them here too, or the tensor count of such a file fails.
    // Absent ones are ignored. (e20efe765, kept over upstream's loader)
    if (!ml.load_mtp) {
        const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;
        for (int il = n_layer; il < n_layer_all; ++il) {
            for (const char * suffix : { "scale", "input_scale" }) {
                for (const llm_tensor t : { LLM_TENSOR_ATTN_Q, LLM_TENSOR_ATTN_K, LLM_TENSOR_ATTN_V, LLM_TENSOR_ATTN_OUT,
                        LLM_TENSOR_ATTN_QKV, LLM_TENSOR_ATTN_GATE, LLM_TENSOR_FFN_GATE_SHEXP, LLM_TENSOR_FFN_DOWN_SHEXP,
                        LLM_TENSOR_FFN_UP_SHEXP, LLM_TENSOR_SSM_OUT, LLM_TENSOR_SSM_ALPHA, LLM_TENSOR_SSM_BETA,
                        LLM_TENSOR_NEXTN_EH_PROJ, LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD }) {
                    create_tensor(tn(t, suffix, il), { 1 }, skip);
                }
                for (const llm_tensor t : { LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_DOWN_EXPS, LLM_TENSOR_FFN_UP_EXPS }) {
                    create_tensor(tn(t, suffix, il), { n_expert }, skip);
                }
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// [TAG_FN_L3_GPU] the lever switches, through the qwen4exp profile's lookups (llama_model_fn_env: the environment first)
llama_model_qwen4exp::graph::l3_flags llama_model_qwen4exp::graph::l3_read(const llama_model & model) {
    const char * all_v = llama_model_fn_env(&model, "LLAMA_FN_GPU");
    const bool   all   = all_v && all_v[0] && all_v[0] != '0';
    auto on = [&](const char * name) {
        const char * v = llama_model_fn_env(&model, name);
        return v && v[0] ? v[0] != '0' : all;
    };
    l3_flags f;
    f.defer   = on("LLAMA_FN_GPU_DEFER");
    f.convwb  = on("LLAMA_FN_GPU_CONVWB");
    f.hcfuse  = on("LLAMA_FN_GPU_HCFUSE");
    f.compact = on("LLAMA_FN_GPU_COMPACT");
    f.topk    = on("LLAMA_FN_GPU_TOPK");
    f.idxq8   = on("LLAMA_FN_GPU_IDXQ8");
    f.mmv     = on("LLAMA_FN_GPU_MMV");
    f.mmvd    = on("LLAMA_FN_GPU_MMVD");
    f.q8f     = on("LLAMA_FN_GPU_Q8F");

    const int mask = (f.defer ? 1 : 0) | (f.convwb ? 2 : 0) | (f.hcfuse ? 4 : 0) | (f.compact ? 8 : 0) |
                     (f.topk ? 16 : 0) | (f.idxq8 ? 32 : 0) | (f.mmv ? 64 : 0) | (f.mmvd ? 128 : 0) | (f.q8f ? 256 : 0);
    static std::atomic<int> logged{-1};
    if (logged.exchange(mask) != mask) {
        LLAMA_LOG_INFO("qwen4exp: [TAG_FN_L3_GPU] device levers:%s%s%s%s%s%s%s%s%s%s\n",
                f.defer ? " DEFER" : "", f.convwb ? " CONVWB" : "", f.hcfuse ? " HCFUSE" : "", f.compact ? " COMPACT" : "",
                f.topk ? " TOPK" : "", f.idxq8 ? " IDXQ8" : "", f.mmv ? " MMV" : "", f.mmvd ? " MMVD" : "",
                f.q8f ? " Q8F" : "", mask == 0 ? " none" : "");
    }
    return f;
}

void llama_model_qwen4exp::graph::l3_expand_deferred() {
    for (ggml_tensor * t : l3_deferred) {
        ggml_build_forward_expand(gf, t);
    }
    l3_deferred.clear();
}

// [TAG_FN_L3_GPU_MMV] marks a plain mat-vec node (a LoRA or a scale tensor makes build_lora_mm return another op)
static void qwen4exp_l3_mark_mm(ggml_tensor * t, bool on) {
    if (on && t && t->op == GGML_OP_MUL_MAT) {
        ggml_fn_l3_set(t, GGML_FN_L3_MMV);
    }
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    if (l3.q8f) {
        ggml_fn_l3_set(xn, GGML_FN_L3_Q8OUT); // [TAG_FN_L3_GPU_Q8F] the q8_1 copy for w_down, made by the norm kernel
    }
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    qwen4exp_l3_mark_mm(lo, l3.mmv); // [TAG_FN_L3_GPU_MMV] 320 rows of 10240
    ggml_tensor * lo_s = ggml_scale(ctx0, lo, 1.0f / (float) hc);
    if (l3.hcfuse) {
        ggml_fn_l3_set(lo_s, GGML_FN_L3_HCLO); // [TAG_FN_L3_GPU_HCFUSE] scale -> silu in one CUDA launch
    }
    lo = ggml_silu(ctx0, lo_s);
    if (l3.q8f && l3.hcfuse) {
        ggml_fn_l3_set(lo, GGML_FN_L3_Q8OUT); // [TAG_FN_L3_GPU_Q8F] the fused scale -> silu also writes the copy for w_up
    }
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    qwen4exp_l3_mark_mm(gate, l3.mmv); // [TAG_FN_L3_GPU_MMV] 10240 rows of 320: one warp per block
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
        if (l3.q8f) {
            ggml_fn_l3_set(mixed, GGML_FN_L3_Q8OUT); // [TAG_FN_L3_GPU_Q8F] the copy for the block's projections / experts
        }
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        qwen4exp_l3_mark_mm(*inject, l3.mmv); // [TAG_FN_L3_GPU_MMV] 4 rows of 10240
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    return build_hc_combine_post(residual, block_out, build_hc_combine_w(inject, il), il);
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine_w(
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w_s = ggml_scale(ctx0, inject, 1.0f / (float) hc);
    if (l3.hcfuse && il >= 0) {
        ggml_fn_l3_set(w_s, GGML_FN_L3_HCW); // [TAG_FN_L3_GPU_HCFUSE] scale -> sigmoid -> scale (-> hc post) in one launch
    }
    ggml_tensor * w = ggml_sigmoid(ctx0, w_s);
    w = ggml_scale(ctx0, w, 2.0f);
    return w;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine_post(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * w,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), l3(l3_read(model)), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // [TAG_FN_PLE_HOST_GATHER] host-side token rows when enabled (build_inp_embd applies no scale or pad for this arch)
    ggml_tensor * inpL = hparams.n_embd_inp() == hparams.n_embd && hparams.f_embedding_scale == 0.0f
        ? build_inp_embd_host(model.tok_embd, nullptr) : nullptr;
    if (inpL) {
        res->t_inp_embd = inpL;
    } else {
        inpL = build_inp_embd(model.tok_embd);
    }
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    // the QSA layers share one set of k-pool inputs
    // the CUDA lightning indexer takes 32 or 64 heads, QSA has a few, so it scores with plain ops
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_idx && hparams.indexer_kpool > 0) {
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
    // make sure hc_init is in the same graph split as the first layer (-sm tensor)
    ggml_build_forward_expand(gf, res_hc);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        // [TAG_FN_L3_GPU_DEFER] the ffn half's combine weights read only this mixer's norm: build_layer_ffn puts them
        // into the graph after the bridge post, so the device computes them while the host runs the experts
        ggml_tensor * w_ffn = nullptr;
        if (l3.defer) {
            w_ffn = build_hc_combine_w(inject, il);
            l3_deferred.push_back(w_ffn);
        }

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = w_ffn ? build_hc_combine_post(res_hc, cur, w_ffn, il) : build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    l3_expand_deferred(); // [TAG_FN_L3_GPU_DEFER] nothing is left here; build_layer_ffn takes them

    // the MTP head reads the hc-wide residual, before the final mixer
    if (cparams.embeddings_nextn) {
        res->t_h_nextn = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        cb(res->t_h_nextn, "h_nextn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
    }

    if (cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && inp_out_ids) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// [TAG_FN_MTP_HEAD_ROWS] a constant I32 input (the head's extra row ids)
class llm_graph_input_ids_const : public llm_graph_input_i {
public:
    llm_graph_input_ids_const(const std::vector<int32_t> & v) : v(v) {}
    virtual ~llm_graph_input_ids_const() = default;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        ggml_backend_tensor_set(ids, v.data(), 0, v.size()*sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return true;
    }

    ggml_tensor * ids = nullptr;

    const std::vector<int32_t> & v;
};

// [TAG_FN_MTP_HEAD_IDS] the compact draft head
llama_model_qwen4exp::mtp_head_compact::~mtp_head_compact() {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

const llama_model_qwen4exp::mtp_head_compact * llama_model_qwen4exp::mtp_head_get(const ggml_tensor * head_w) const {
    std::lock_guard<std::mutex> lock(mtp_head_mutex);
    if (mtp_head_tried || mtp_head_ids.empty()) {
        return mtp_head_c.get();
    }
    // a model loaded without tensor data (a memory-fit probe) has nothing to copy yet: ask again next graph
    if (head_w == nullptr || head_w->buffer == nullptr || head_w->data == nullptr) {
        return nullptr;
    }
    mtp_head_tried = true;

    const int64_t n_ids     = (int64_t) mtp_head_ids.size();
    const size_t  row_bytes = ggml_row_size(head_w->type, head_w->ne[0]);
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(head_w->buffer);
    ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);

    // rows must be plain bytes: a device's own buffer type or the CPU one (not a repacked or split layout)
    const bool plain = buft == ggml_backend_cpu_buffer_type() || (dev != nullptr && ggml_backend_dev_buffer_type(dev) == buft);
    if (!plain || head_w->nb[1] != row_bytes || !ggml_is_contiguous(head_w) ||
            split_mode() == LLAMA_SPLIT_MODE_ROW || split_mode() == LLAMA_SPLIT_MODE_TENSOR ||
            mtp_head_ids.back() >= head_w->ne[1]) {
        LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_IDS] head in '%s' (%lld rows) cannot be copied row by row: full head\n",
                __func__, ggml_backend_buft_name(buft), (long long) head_w->ne[1]);
        return nullptr;
    }

    // gather the rows through a bounded host window; the ids are sorted, so the head is read once
    std::vector<uint8_t> rows((size_t) n_ids*row_bytes);
    {
        const int64_t win = std::max<int64_t>(1, (int64_t) ((64u << 20)/row_bytes));
        std::vector<uint8_t> chunk((size_t) std::min<int64_t>(win, head_w->ne[1])*row_bytes);
        int64_t k = 0;
        for (int64_t r0 = 0; r0 < head_w->ne[1] && k < n_ids; r0 += win) {
            const int64_t r1 = std::min<int64_t>(head_w->ne[1], r0 + win);
            if (mtp_head_ids[k] >= r1) {
                continue;
            }
            ggml_backend_tensor_get(head_w, chunk.data(), (size_t) r0*row_bytes, (size_t) (r1 - r0)*row_bytes);
            for (; k < n_ids && mtp_head_ids[k] < r1; ++k) {
                memcpy(rows.data() + (size_t) k*row_bytes, chunk.data() + (size_t) (mtp_head_ids[k] - r0)*row_bytes, row_bytes);
            }
        }
        GGML_ASSERT(k == n_ids);
    }

    auto c = std::make_unique<mtp_head_compact>();
    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    c->ctx = ggml_init(ip);
    c->w   = ggml_new_tensor_2d(c->ctx, head_w->type, head_w->ne[0], n_ids);
    c->ids = ggml_new_tensor_1d(c->ctx, GGML_TYPE_I32, n_ids);
    ggml_set_name(c->w,   "mtp_head_compact.weight");
    ggml_set_name(c->ids, "mtp_head_compact.ids");
    c->buf = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx, buft);
    if (c->buf == nullptr) {
        LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_IDS] cannot allocate %.1f MiB in '%s': full head\n",
                __func__, rows.size()/1048576.0, ggml_backend_buft_name(buft));
        return nullptr;
    }
    ggml_backend_buffer_set_usage(c->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(c->w,   rows.data(),         0, rows.size());
    ggml_backend_tensor_set(c->ids, mtp_head_ids.data(), 0, (size_t) n_ids*sizeof(int32_t));

    LLAMA_LOG_INFO("%s: [TAG_FN_MTP_HEAD_IDS] compact MTP draft head: %lld of %lld rows, %.1f MiB %s in '%s'\n",
            __func__, (long long) n_ids, (long long) head_w->ne[1], ggml_backend_buffer_get_size(c->buf)/1048576.0,
            ggml_type_name(head_w->type), ggml_backend_buft_name(buft));
    mtp_head_c = std::move(c);
    return mtp_head_c.get();
}

llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP has a single block");
    GGML_ASSERT(ubatch.token && "qwen4exp MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "qwen4exp MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.hc_head_norm &&
            "MTP block missing, load the model with MTP enabled");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // [TAG_FN_PLE_HOST_GATHER] host-side token rows: the draft graph then starts on the GPU too
    ggml_tensor * h        = nullptr;
    ggml_tensor * tok_embd = build_inp_embd_host(model.tok_embd, &h);
    if (!tok_embd) {
        auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tokens);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
        ggml_set_input(inp->embd);

        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
        ggml_set_input(inp->h);
        ggml_set_name(inp->h, "mtp_h_input");

        tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
        h        = inp->h;

        res->add_input(std::move(inp));
    }
    cb(tok_embd, "mtp_tok_embd", il);

    auto * inp_hyb = build_inp_mem_hybrid();
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);

    // the draft memory has no recurrent layer, but its input still has to be allocated
    ggml_build_forward_expand(gf, inp_hyb->get_recr()->s_copy);

    // [TAG_QWEN4EXP_MTP] [TAG_SYNC_1004] the indexer inputs only for a QSA block: an MTP block with compress ratio 0 runs
    // dense (build_layer_attn), and an input no node reads is never allocated, so its set_input would assert. The
    // unsloth Qwen3.8-Flash-Next MTP files carry ratio 0 for the nextn block.
    llm_graph_input_kpool * inp_kpool = nullptr;
    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0 && hparams.dsv4_compress_ratios[il] > 0) {
        GGML_ASSERT(mctx_hyb->get_idx()->get_n_kv() == mctx_hyb->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
        inp_kpool = build_inp_kpool(mctx_hyb);
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * h_norm = build_norm(ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens), layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0)); // [n_embd, hc, n_tokens]
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
    cur    = build_layer_attn(inp_hyb->get_attn(), mctx_hyb, inp_kpool, cur, inp_pos, sections, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    cur    = build_hc_mix(res_hc, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
    cur    = build_layer_ffn(cur, il);
    res_hc = build_hc_combine(res_hc, cur, inject, il);

    // the next draft step reads this residual as its h
    ggml_tensor * flat     = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;
    res->t_h_nextn = cparams.embeddings_nextn_masked ? flat_out : flat;
    cb(res->t_h_nextn, "h_nextn", il);
    ggml_build_forward_expand(gf, res->t_h_nextn);

    cur = build_hc_mix(ggml_reshape_3d(ctx0, flat_out, n_embd, hc, flat_out->ne[1]),
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, il);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = model.output;
    ggml_tensor * head_s = model.output_s;

    // [TAG_FN_MTP_HEAD_ROWS] rows [0, N) of the head as a contiguous view, the extra ids by get_rows, -inf elsewhere:
    // the logits keep the full vocabulary width, so the draft driver and backend sampling are unchanged
    const auto & pm = static_cast<const llama_model_qwen4exp &>(model);
    const int64_t n_vocab_head = head_w->ne[1];

    // [TAG_FN_MTP_HEAD_IDS] a per-tensor head scale (one value) multiplies any subset of rows the same way
    const bool head_s_scalar = head_s == nullptr || ggml_nelements(head_s) == 1;

    // [TAG_FN_MTP_HEAD_IDS] the calibrated draft vocabulary: its byte-copied rows, scattered into a -inf row of the full
    // width (the draft driver and backend sampling see the usual logits)
    const llama_model_qwen4exp::mtp_head_compact * head_c = nullptr;
    if (!pm.mtp_head_ids.empty() && head_s_scalar && loras->empty() && n_vocab_head == (int64_t) model.vocab.n_tokens()) {
        head_c = pm.mtp_head_get(head_w);
    }

    if (head_c != nullptr) {
        const int64_t n_ids = head_c->w->ne[1];
        const int64_t n_out = cur->ne[1];

        ggml_tensor * draft = ggml_mul_mat(ctx0, head_c->w, cur);
        if (head_s) {
            draft = ggml_mul(ctx0, draft, head_s);
        }
        cur = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_head, n_out), -INFINITY);
        cur = ggml_set_rows(ctx0, cur,
                ggml_reshape_3d(ctx0, draft, 1, n_ids, n_out),
                ggml_reshape_2d(ctx0, head_c->ids, n_ids, 1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab_head, n_out);
    } else if (pm.mtp_head_rows > 0 && pm.mtp_head_rows < n_vocab_head && head_s_scalar && loras->empty() &&
            n_vocab_head == (int64_t) model.vocab.n_tokens()) {
        const int64_t n_rows  = pm.mtp_head_rows;
        const int64_t n_extra = (int64_t) pm.mtp_head_extra.size();
        const int64_t n_out   = cur->ne[1];
        ggml_tensor * normed  = cur;

        cur = ggml_mul_mat(ctx0, ggml_view_2d(ctx0, head_w, head_w->ne[0], n_rows, head_w->nb[1], 0), normed);
        if (head_s) {
            cur = ggml_mul(ctx0, cur, head_s); // [TAG_FN_MTP_HEAD_IDS] the per-tensor scale, as build_lora_mm applies it
        }
        cur = ggml_concat(ctx0, cur,
                ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab_head - n_rows, n_out), -INFINITY), 0);
        if (n_extra > 0) {
            auto inp_ids = std::make_unique<llm_graph_input_ids_const>(pm.mtp_head_extra);
            inp_ids->ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_extra);
            ggml_set_input(inp_ids->ids);

            ggml_tensor * extra = ggml_mul_mat(ctx0, ggml_get_rows(ctx0, head_w, inp_ids->ids), normed);
            if (head_s) {
                extra = ggml_mul(ctx0, extra, head_s);
            }
            cur = ggml_set_rows(ctx0, ggml_reshape_3d(ctx0, cur, 1, n_vocab_head, n_out),
                    ggml_reshape_3d(ctx0, extra, 1, n_extra, n_out),
                    ggml_reshape_2d(ctx0, inp_ids->ids, n_extra, 1));
            cur = ggml_reshape_2d(ctx0, cur, n_vocab_head, n_out);

            res->add_input(std::move(inp_ids));
        }
    } else {
        if ((pm.mtp_head_rows > 0 || !pm.mtp_head_ids.empty()) && head_w->buffer != nullptr) {
            // [TAG_FN_MTP_HEAD_ROWS] [TAG_FN_MTP_HEAD_IDS] say so once, so a draft-vocabulary arm is not read as "no effect"
            static bool warned = false;
            if (!warned) {
                warned = true;
                LLAMA_LOG_WARN("%s: [TAG_FN_MTP_HEAD_ROWS] draft vocabulary not used: head %lld rows vs vocab %u, scale %s, %zu LoRA - full head\n",
                        __func__, (long long) n_vocab_head, model.vocab.n_tokens(), head_s ? (head_s_scalar ? "scalar" : "per row") : "none", loras->size());
            }
        }
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qwen4exp_l3_mark_mm(qkv_mixed, l3.mmvd); // [TAG_FN_L3_GPU_MMV]
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    qwen4exp_l3_mark_mm(z, l3.mmvd);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA k-pool inputs, shared by the QSA layers: blocks of compress_ratio cells in sequence order, see llama_memory_hybrid_idx
class llama_model_qwen4exp::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t kpool) : mctx(mctx), kpool(kpool) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_kpool(pool_cells, pool_idxs, pool_mask, tail_idxs, nullptr, false, new_pool_idxs, new_pool_rep,
                              ubatch, new_pool_pos);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0]     == params.ubatch.n_tokens;
        res &= pool_cells->ne[0] == mctx->get_n_kpool();
        res &= pool_mask->ne[1]  == params.ubatch.n_tokens;
        res &= tail_idxs->ne[1]  == params.ubatch.n_tokens;
        // the scatter mask shape follows n_kv
        res &= n_kv              == idx->get_n_kv();
        res &= n_new             == mctx->get_n_kpool_new();
        res &= cache_safe        == mctx->get_kpool_cache_safe();

        return res;
    }

    ggml_tensor * k_idxs        = nullptr; // I64 [n_tokens]
    ggml_tensor * pool_cells    = nullptr; // I32 [n_pool]         cell caching each block's pooled key
    ggml_tensor * pool_idxs     = nullptr; // I32 [kpool, n_pool]  member cells per block, n_kv sentinel for the padded blocks
    ggml_tensor * pool_mask     = nullptr; // F16 [n_pool, n_tokens]
    ggml_tensor * tail_idxs     = nullptr; // I32 [kpool - 1, n_tokens]
    ggml_tensor * new_pool_idxs = nullptr; // I32 [kpool, n_new]   members of the blocks to re-pool this ubatch
    ggml_tensor * new_pool_rep  = nullptr; // I64 [n_new]          cell to write each new pooled key into
    ggml_tensor * new_pool_pos  = nullptr; // I32 [4*n_new]        M-RoPE position of each new block's first member

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t kpool;
    uint32_t n_new = 0; // padded to a stable bound, never below 1
    uint32_t n_sel = 0;
    uint32_t n_kv  = 0;
    bool cache_safe = true;
};

llama_model_qwen4exp::llm_graph_input_kpool * llama_model_qwen4exp::graph::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool  = hparams.indexer_kpool;
    const uint32_t n_pool = mctx_hyb->get_n_kpool();

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, kpool);

    inp->k_idxs     = mctx_idx->build_input_k_idxs(ctx0, ubatch);
    inp->pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
    inp->pool_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, n_pool);
    inp->pool_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n_tokens);
    inp->tail_idxs  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
    ggml_set_input(inp->pool_cells);
    ggml_set_input(inp->pool_idxs);
    ggml_set_input(inp->pool_mask);
    ggml_set_input(inp->tail_idxs);

    // set_input fills them all, so keep them allocated even when no op reads them
    ggml_build_forward_expand(gf, inp->pool_cells);
    ggml_build_forward_expand(gf, inp->pool_idxs);
    ggml_build_forward_expand(gf, inp->pool_mask);
    ggml_build_forward_expand(gf, inp->tail_idxs);

    inp->n_kv       = mctx_idx->get_n_kv();
    inp->n_new      = mctx_hyb->get_n_kpool_new();
    inp->cache_safe = mctx_hyb->get_kpool_cache_safe();
    // the top blocks plus the tail
    inp->n_sel      = kpool*std::min<uint32_t>(n_pool, hparams.indexer_top_k / kpool) + kpool - 1;

    inp->new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_new);
    ggml_set_input(inp->new_pool_idxs);
    if (inp->cache_safe) {
        inp->new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, inp->n_new);
        ggml_set_input(inp->new_pool_rep);
    }
    inp->new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*inp->n_new);
    ggml_set_input(inp->new_pool_pos);

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

// QSA attends to the top blocks of compress_ratio cells plus the incomplete tail, like the glm5-next k-pool indexer
// a block is scored by one pooled key: the mean of its raw indexer keys, normed and rotated to its first member
llama_model_qwen4exp::graph::qsa_sel llama_model_qwen4exp::graph::build_qsa_sel(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *                 inp_kpool,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        llm_graph_input_attn_kv *               inp,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    ggml_tensor * kq_mask = inp->get_kq_mask(); // [TAG_FN_R4_QSA_POS] nullptr with the positional vectors

    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t n_idx_h = hparams.indexer_n_head;
    const int64_t kpool   = inp_kpool->kpool;
    const int64_t n_pool  = inp_kpool->pool_cells->ne[0];
    const int64_t n_new   = inp_kpool->n_new;

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == kpool);

    // cache rows store raw key | pooled key: pooling precedes norm and rotation, so the raw key gets neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    cb(k_raw, "indexer_k_raw", il);

    ggml_tensor * pzero  = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_tokens), 0.0f);
    ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2*idx_dim, 1, n_tokens);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));

    // the raw keys and the persistent pooled slots, see llama_memory_hybrid_idx::mem_idx_stale
    auto kpool_cache = mctx_hyb->get_kpool_access(ctx0, il, idx_dim);

    // pool only the blocks this ubatch completes or regroups
    ggml_tensor * rows = kpool_cache.gather_key_gate(ggml_reshape_1d(ctx0, inp_kpool->new_pool_idxs, kpool*n_new));
    rows = ggml_reshape_3d(ctx0, rows, idx_dim, kpool, n_new);

    // mean over the members; kpool is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled_new = nullptr;
    for (int64_t i = 0; i < kpool; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx0, rows, idx_dim, n_new, rows->nb[2], i*rows->nb[1]);
        pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
    }
    pooled_new = ggml_scale(ctx0, pooled_new, 1.0f/(float) kpool);
    pooled_new = build_norm(pooled_new, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    pooled_new = ggml_reshape_3d(ctx0, pooled_new, idx_dim, 1, n_new);
    pooled_new = ggml_rope_multi(ctx0, pooled_new, inp_kpool->new_pool_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled_new = ggml_reshape_2d(ctx0, pooled_new, idx_dim, n_new);
    cb(pooled_new, "indexer_pool_k_new", il);

    ggml_tensor * pooled = nullptr;
    if (inp_kpool->cache_safe) {
        // write before the pool gather
        ggml_build_forward_expand(gf, kpool_cache.scatter_pooled(pooled_new, inp_kpool->new_pool_rep));
        pooled = kpool_cache.gather_pooled(inp_kpool->pool_cells);
    } else {
        // shared cells re-pool every pool, in layout order
        GGML_ASSERT(n_new < n_pool);
        ggml_tensor * pad = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, idx_dim, n_pool - n_new), 0.0f);
        pooled = ggml_concat(ctx0, pooled_new, pad, 1);
    }
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_pool);
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, hparams.indexer_top_k / kpool);

    // [TAG_TOPK_UNORDERED] the picked pools only feed the gathers and the ggml_set_rows below that unmask the cells
    // they name, so their order is never read. Unordered lets CUDA radix-select instead of sorting the whole score
    // row. Ties on the k-th value keep the lowest indices, the same set a stable sort gives.
    // TURBO_QSA_TOPK_UNORDERED=1 turns it on (default off until measured); TURBO_TOPK_ORDERED=1 overrides it in CUDA.
    // [TAG_SYNC_1004] upstream #29751/#29825 moved the selection to the pooled lightning-indexer score.
    static const bool topk_unordered = [] {
        const char * e = getenv("TURBO_QSA_TOPK_UNORDERED");
        return e && e[0] == '1';
    }();

    // the scores of nt queries and their picks: q_part [idx_dim, n_idx_h, nt], mask_part [n_pool, nt]; top_score gets
    // the picked scores [1, n_top_pool, nt]
    auto score_top_k = [&](ggml_tensor * q_part, ggml_tensor * mask_part, int64_t nt, ggml_tensor ** top_score) {
        // the reference sums the rectified head scores unweighted, scaled by 1/sqrt(head_dim),
        // which is the lightning indexer with every head weight set to that scale
        ggml_tensor * weights = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_idx_h, nt), 1.0f/sqrtf((float) idx_dim));
        ggml_tensor * score = ggml_lightning_indexer(ctx0, q_part, pooled, weights, mask_part); // [n_pool, nt]
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
        cb(score, "indexer_score", il);
        if (l3.idxq8) {
            // [TAG_FN_L3_GPU_IDXQ8] CUDA reads the pooled keys from the q8_0 cache rows through the gather's indices
            // (skipping the f32 gather when this is its only reader), with the dequantization the gather would do
            ggml_fn_l3_set(score, GGML_FN_L3_IDXQ8);
        }

        ggml_tensor * top_k = topk_unordered ? ggml_top_k_unordered(ctx0, score, n_top_pool)
                                             : ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, nt], unordered
        cb(top_k, "indexer_top_k", il);
        if (l3.topk && topk_unordered) {
            ggml_fn_l3_set(top_k, GGML_FN_L3_TOPK); // [TAG_FN_L3_GPU_TOPK] the same set from a chunked two-stage select
        }

        *top_score = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, score, 1, n_pool, nt), top_k); // [1, n_top_pool, nt]
        return top_k;
    };

    // [TAG_FN_QSA_CHUNK] TURBO_QSA_CHUNK=<n>: score and pick for n queries at a time, so the f32 [n_pool, n_tokens] scores
    // and the indexer temporaries shrink to [n_pool, n]. The picks are per query, so they are the same; the chunks'
    // picks and picked scores are concatenated for the unchanged mask below. 0 (default) = off. [TAG_SYNC_1004] upstream
    // #29751/#29825 score pools, not cells, so the [n_kv, n_ubatch] f32 temporaries this lever first cut are gone.
    static const int64_t qsa_chunk = [] {
        const char * e = getenv("TURBO_QSA_CHUNK");
        return e ? std::max<int64_t>(0, atoll(e)) : 0;
    }();

    ggml_tensor * top_k     = nullptr;
    ggml_tensor * top_score = nullptr;
    if (qsa_chunk > 0 && n_tokens > qsa_chunk) {
        ggml_tensor * pool_mask = inp_kpool->pool_mask;
        for (int64_t c0 = 0; c0 < n_tokens; c0 += qsa_chunk) {
            const int64_t nc = std::min<int64_t>(qsa_chunk, n_tokens - c0);
            ggml_tensor * q_c    = ggml_view_3d(ctx0, q, idx_dim, n_idx_h, nc, q->nb[1], q->nb[2], c0*q->nb[2]);
            ggml_tensor * mask_c = ggml_view_2d(ctx0, pool_mask, n_pool, nc, pool_mask->nb[1], c0*pool_mask->nb[1]);
            ggml_tensor * ts_c   = nullptr;
            ggml_tensor * tk_c   = score_top_k(q_c, mask_c, nc, &ts_c);
            top_k     = top_k     ? ggml_concat(ctx0, top_k,     tk_c, 1) : tk_c;
            top_score = top_score ? ggml_concat(ctx0, top_score, ts_c, 2) : ts_c;
        }
    } else {
        top_k = score_top_k(q, inp_kpool->pool_mask, n_tokens, &top_score);
    }

    // the top blocks, then the incomplete tail with n_kv for missing cells
    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens)); // [kpool, n_top_pool*n_tokens]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);
    sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
    const int64_t n_sel = sel_idx->ne[0];
    GGML_ASSERT(n_sel == inp_kpool->n_sel);

    ggml_build_forward_expand(gf, sel_idx);

    // [TAG_FN_R4_QSA_POS] the positional vectors: no explicit mask to add; ggml_qsa_mask builds the mask next to the
    // attention from the selection, the picked scores (a picked pool is live when its score is not -inf, the rule below)
    // and the positions. Its values are the ones the scatter below gives.
    if (kq_mask == nullptr) {
        GGML_ASSERT(inp->get_kv_pos() != nullptr && inp->get_q_pos() != nullptr);
        qsa_sel qs;
        qs.sel_idx = sel_idx;
        qs.live    = ggml_reshape_2d(ctx0, top_score, n_top_pool, n_tokens);
        qs.group   = kpool;
        qs.n_sel   = n_sel;
        cb(qs.live, "indexer_top_score", il);
        return qs;
    }

    // TODO: figure out to reduce the large copmute buffer that this creates

    // scatter zeros for the selected cells into an all -inf row, each dead slot into its own dump row n_kv + slot
    // seeding from sel_idx ties the scatter storage lifetime to this layer
    const int64_t n_kv = inp_kpool->n_kv;

    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, kq_mask->type, n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    // live slots address disjoint cells, but padded pools and missing tail cells share the n_kv sentinel, and
    // top_k fills a short selection with invisible pools that can overlap the tail, so the scatter would write
    // some cells from several threads: map every dead slot to its own dump row, idx = dump + live*(idx - dump)
    // a picked pool is live when visible: a visible score is a rectified sum >= 0, an invisible one is -inf
    // top_score: the picked scores [1, n_top_pool, n_tokens], from score_top_k
    ggml_tensor * live_pool = ggml_clamp(ctx0, ggml_scale_bias(ctx0, top_score, 1.0f, 1.0f), 0.0f, 1.0f);
    live_pool = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, live_pool, kpool, n_top_pool, n_tokens, 1), kpool*n_top_pool, n_tokens);
    // a tail cell is live unless it is the n_kv sentinel
    ggml_tensor * live_tail = ggml_cast(ctx0, inp_kpool->tail_idxs, GGML_TYPE_F32);
    live_tail = ggml_clamp(ctx0, ggml_scale_bias(ctx0, live_tail, -1.0f, (float) n_kv), 0.0f, 1.0f);
    ggml_tensor * live = ggml_concat(ctx0, live_pool, live_tail, 0); // [n_sel, n_tokens]

    // dump rows n_kv + slot as a cumulative sum: the meta backend cannot split an arange, which has no source
    ggml_tensor * dump  = ggml_scale_bias(ctx0, ggml_cumsum(ctx0, ggml_fill(ctx0, live, 1.0f)), 1.0f, (float) (n_kv - 1));
    ggml_tensor * idx_f = ggml_cast(ctx0, sel_idx, GGML_TYPE_F32);
    idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);
    sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));

    GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
            row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
    sel = ggml_add(ctx0, sel, kq_mask);
    cb(sel, "indexer_sel", il);

    qsa_sel qs;
    qs.mask  = sel;
    qs.group = kpool;
    qs.n_sel = n_sel;
    return qs;
}

// [TAG_FN_L3_GPU_COMPACT] marks the FLASH_ATTN_EXT node under the reshape / inverse WHT that build_attn_mha_kv puts on it
static void qwen4exp_l3_mark_fa(ggml_tensor * t, bool on) {
    for (int i = 0; on && t != nullptr && i < 8; ++i) {
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_fn_l3_set(t, GGML_FN_L3_COMPACT);
            return;
        }
        if (t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && t->op != GGML_OP_PERMUTE &&
            t->op != GGML_OP_CONT && t->op != GGML_OP_TURBO_WHT) {
            return;
        }
        t = t->src[0];
    }
}

// Dense GQA self-attention over the cells that the QSA mask keeps.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        const qsa_sel &           qs,
        float                     kq_scale,
        int                       il) {
    const int64_t n_sel = qs.n_sel;
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_sel, so the selection is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        if (mctx_cur->is_turbot()) {
            // [TAG_FN_TURBOT_QSA] the turbot writer, as in llm_graph_context::build_attn: base code, the young rows of
            // young granules and the center fills of this ubatch
            ggml_build_forward_expand(gf, mctx_cur->turbot_cpy_k(ctx0, k_cur, k_idxs, inp->self_turbot_young, inp->self_turbot_fill, il));
            ggml_build_forward_expand(gf, mctx_cur->turbot_cpy_v(ctx0, v_cur, v_idxs, inp->self_turbot_young, inp->self_turbot_fill, il));
        } else {
            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
        }
    }

    // the selection mask already carries the causal mask
    ggml_tensor * mask = nullptr;
    if (qs.mask) {
        ggml_tensor * kq_mask = inp->get_kq_mask();
        mask = ggml_reshape_4d(ctx0, qs.mask, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);
        cb(mask, "kq_mask_qsa", il);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // [TAG_FN_TURBOT_QSA] the read side of the dense path: the forward WHT on Q for a turbo or turbot K (12260c288, lost
    // in the 09-21 upstream sync), and for turbot the young pool, the granule table and the op params. f16 and q8_0 get
    // the same nodes as upstream's build_attn_mha with the selection budget n_sel.
    // [TAG_FN_TURBOT_SPARSE] TURBO_QSA_SPARSE=1: a turbot FA gets the budget as n_kv_max too, so CUDA gathers only the
    // selected cells (at most n_sel per query) instead of reading every cell under the mask. Default off: the turbot
    // FA gets n_kv_max 0 and runs dense (f16 keeps its own gather either way).
    static const bool qsa_sparse_turbot = [] {
        const char * e = getenv("TURBO_QSA_SPARSE");
        return e && e[0] == '1';
    }();
    const int64_t n_kv_max = !mctx_cur->is_turbot() || qsa_sparse_turbot ? n_sel : 0;
    ggml_tensor * cur = nullptr;
    if (mask) {
        cur = build_attn_mha_kv(mctx_cur, q, k, v, nullptr, mask, nullptr, nullptr, kq_scale, il,
                nullptr, nullptr, inp->self_turbot_gtab, n_kv_max);
        qwen4exp_l3_mark_fa(cur, l3.compact);
    } else {
        // [TAG_FN_R4_QSA_POS] the mask from the positional vectors, the selection and the picked scores: one f16 [n_kv, n]
        // buffer instead of the explicit KQ mask input, the -inf fill, the scatter and the sum (each about n_kv x n_ubatch).
        // LLAMA_QSA_POS_CHUNK=<n>: the mask and the attention per n queries (the outputs are concatenated in query order),
        // so the buffer is [n_kv, n]; every chunk reads the whole cache again.
        ggml_tensor * kv_pos = inp->get_kv_pos();
        ggml_tensor * q_pos  = inp->get_q_pos();
        GGML_ASSERT(kv_pos && q_pos && qs.sel_idx && qs.live);
        const int64_t n_q   = q->ne[2];
        const int64_t chunk = cparams.qsa_pos_chunk > 0 ? (int64_t) cparams.qsa_pos_chunk : n_q;
        for (int64_t c0 = 0; c0 < n_q; c0 += chunk) {
            const int64_t nc = std::min(chunk, n_q - c0);
            ggml_tensor * q_c  = nc == n_q ? q : ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], nc, q->nb[1], q->nb[2], c0*q->nb[2]);
            ggml_tensor * qp_c = nc == n_q ? q_pos : ggml_view_2d(ctx0, q_pos, nc, q_pos->ne[1], q_pos->nb[1], c0*q_pos->nb[0]);
            ggml_tensor * sl_c = nc == n_q ? qs.sel_idx :
                ggml_view_2d(ctx0, qs.sel_idx, qs.sel_idx->ne[0], nc, qs.sel_idx->nb[1], c0*qs.sel_idx->nb[1]);
            ggml_tensor * lv_c = nc == n_q ? qs.live :
                ggml_view_2d(ctx0, qs.live, qs.live->ne[0], nc, qs.live->nb[1], c0*qs.live->nb[1]);
            ggml_tensor * mask_c = ggml_qsa_mask(ctx0, kv_pos, qp_c, sl_c, lv_c, (int32_t) qs.group);
            cb(mask_c, "kq_mask_qsa", il);
            ggml_tensor * out_c = build_attn_mha_kv(mctx_cur, q_c, k, v, nullptr, mask_c, nullptr, nullptr, kq_scale, il,
                    nullptr, nullptr, inp->self_turbot_gtab, n_kv_max);
            qwen4exp_l3_mark_fa(out_c, l3.compact); // [TAG_FN_L3_GPU_COMPACT]
            cur = cur ? ggml_concat(ctx0, cur, out_c, 1) : out_c;
        }
    }
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        llm_graph_input_kpool *   inp_kpool,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = inp_kpool != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    qsa_sel qs;
    if (qsa) {
        qs = build_qsa_sel(mctx_hyb, inp_kpool, cur, inp_pos, inp, sections, il);
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    qwen4exp_l3_mark_mm(Qcur_full, l3.mmvd); // [TAG_FN_L3_GPU_MMV]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    qwen4exp_l3_mark_mm(Kcur, l3.mmvd);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    qwen4exp_l3_mark_mm(Vcur, l3.mmvd);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (qsa) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, qs, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    qwen4exp_l3_mark_mm(cur, l3.mmvd); // [TAG_FN_L3_GPU_MMV]
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    qwen4exp_l3_mark_mm(beta, l3.mmv); // [TAG_FN_L3_GPU_MMV] 48 rows of 2560
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    qwen4exp_l3_mark_mm(alpha, l3.mmv); // [TAG_FN_L3_GPU_MMV]
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    qwen4exp_l3_mark_mm(cur, l3.mmvd); // [TAG_FN_L3_GPU_MMV]
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // [TAG_FN_MOE_TRACE] LLAMA_MOE_TRACE_PRED: the next layer's router on this layer's FFN input (prefetch study)
    // [TAG_MOE_PREFETCH] the same lookahead feeds the DMA prefetch in decode graphs (LLAMA_MOE_PREFETCH)
    const int trace_k = llama_moe_trace_pred_k();
    const int pred_k  = trace_k > 0 ? trace_k : (n_tokens <= LLAMA_MOE_DMA_MAX_T ? llama_moe_dma_pred_k() : 0);
    if (pred_k > 0 && il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp) {
        ggml_tensor * lg = build_lora_mm(model.layers[il + 1].ffn_gate_inp, cur);
        ggml_tensor * pr = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, lg, std::min<int>(pred_k, (int) n_expert)));
        ggml_format_name(pr, "moe_trace_pred-%d", il + 1);
        if (trace_k > 0) {
            ggml_set_output(pr);
        }
        ggml_build_forward_expand(gf, pr);
        res->t_moe_pred_next[il + 1] = pr;
    }

    // [TAG_MOE_BRIDGE] with the host bridge, build_moe_ffn posts the host experts and returns a placeholder; the wait
    // comes after the shared expert (build_moe_bridge_finish below), so the device computes it while the host works
    moe_bridge_defer = true;
    moe_router_mark  = l3.mmv ? GGML_FN_L3_MMV : 0; // [TAG_FN_L3_GPU_MMV] the router: 512 rows of 2560
    moe_q8in_mark    = l3.q8f ? GGML_FN_L3_Q8IN : 0; // [TAG_FN_L3_GPU_Q8F] the hot / DMA chains read hc_pre's q8_1 copy
    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    moe_bridge_defer = false;
    moe_router_mark  = 0;
    moe_q8in_mark    = 0;

    // [TAG_FN_L3_GPU_DEFER] after the post (and the hot chain), before the shared expert and the wait
    l3_expand_deferred();

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        qwen4exp_l3_mark_mm(ffn_shexp, l3.mmv); // [TAG_FN_L3_GPU_MMV] the down projection: 2560 rows of 640
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        qwen4exp_l3_mark_mm(shared_gate, l3.mmv); // [TAG_FN_L3_GPU_MMV] one row of 2560
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        moe_out = build_moe_bridge_finish(moe_out, il, ffn_shexp); // [TAG_MOE_BRIDGE] unchanged without a post
        cb(moe_out, "ffn_moe_out", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        moe_out = build_moe_bridge_finish(moe_out, il, nullptr); // [TAG_MOE_BRIDGE]
        cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

// [TAG_FN_PLE_HOST_GATHER] LLAMA_PLE_HOST_GATHER=1: the token embedding and PLE rows are dequantized on the host in
// set_input, with the same to_float the CPU GET_ROWS uses (bitwise equal), and uploaded as f32 graph inputs. The step
// then no longer starts with a CPU split (2 hand-offs). Needs the table in a host buffer. Off by default.
static bool qwen4exp_host_gather_ok(const ggml_tensor * table) {
    static const bool on = [] {
        const char * e = getenv("LLAMA_PLE_HOST_GATHER");
        return e != nullptr && atoi(e) != 0;
    }();
    if (!on || table == nullptr || table->data == nullptr || table->buffer == nullptr ||
        !ggml_backend_buffer_is_host(table->buffer)) {
        return false;
    }
    return table->type == GGML_TYPE_F32 || ggml_get_type_traits(table->type)->to_float != nullptr;
}

static void qwen4exp_rows_to_float(const ggml_tensor * table, const int32_t * rows, int64_t n_rows, float * dst) {
    const int64_t n = table->ne[0];
    const ggml_to_float_t to_float = ggml_get_type_traits(table->type)->to_float;
    for (int64_t r = 0; r < n_rows; ++r) {
        GGML_ASSERT(rows[r] >= 0 && rows[r] < table->ne[1]);
        const char * src = (const char *) table->data + (size_t) rows[r]*table->nb[1];
        if (table->type == GGML_TYPE_F32) {
            memcpy(dst + r*n, src, n*sizeof(float));
        } else {
            to_float(src, dst + r*n, n);
        }
    }
}

// [TAG_FN_PLE_DIRECT_IO] the same rows read from the file (unbuffered, all at once) into raw, then the same to_float:
// bit-identical to qwen4exp_rows_to_float. The mapped table is only the fallback for a failed read.
static void qwen4exp_rows_to_float_dio(llama_ple_dio & dio, const ggml_tensor * table, const int32_t * rows, int64_t n_rows,
        int64_t n_tokens, std::vector<uint8_t> & raw, float * dst) {
    const int64_t n  = table->ne[0];
    const size_t  rb = table->nb[1];
    raw.resize((size_t) n_rows*rb);
    dio.read_rows(rows, n_rows, raw.data(), n_tokens, (const uint8_t *) table->data);
    const ggml_to_float_t to_float = ggml_get_type_traits(table->type)->to_float;
    for (int64_t r = 0; r < n_rows; ++r) {
        const char * src = (const char *) raw.data() + (size_t) r*rb;
        if (table->type == GGML_TYPE_F32) {
            memcpy(dst + r*n, src, n*sizeof(float));
        } else {
            to_float(src, dst + r*n, n);
        }
    }
}

// [TAG_FN_PLE_HOST_GATHER] token rows as an f32 input; h is the MTP driver's hidden state input (nullptr in the trunk)
class llm_graph_input_embd_host : public llm_graph_input_i {
public:
    llm_graph_input_embd_host(const ggml_tensor * table) : table(table) {}
    virtual ~llm_graph_input_embd_host() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->token);
        const int64_t n_tokens = ubatch->n_tokens;
        buf.resize(table->ne[0]*n_tokens);
        qwen4exp_rows_to_float(table, ubatch->token, n_tokens, buf.data());
        ggml_backend_tensor_set(embd, buf.data(), 0, buf.size()*sizeof(float));
        if (h && ubatch->embd) {
            ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*h->ne[0]*sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.token && embd->ne[1] == params.ubatch.n_tokens &&
               (!params.ubatch.embd || (h && h->ne[1] == params.ubatch.n_tokens));
    }

    ggml_tensor * embd = nullptr; // F32 [table->ne[0], n_tokens]
    ggml_tensor * h    = nullptr; // F32 [n_embd_out, n_tokens] or nullptr

    const ggml_tensor * table;
    std::vector<float>  buf;
};

ggml_tensor * llama_model_qwen4exp::graph::build_inp_embd_host(ggml_tensor * table, ggml_tensor ** h_out) {
    if (!ubatch.token || !loras->empty() || !qwen4exp_host_gather_ok(table)) {
        return nullptr;
    }
    auto inp = std::make_unique<llm_graph_input_embd_host>(table);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, table->ne[0], n_tokens);
    ggml_set_input(inp->embd);
    cb(inp->embd, "inp_embd_host", -1);
    if (h_out) {
        inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
        ggml_set_input(inp->h);
        ggml_set_name(inp->h, "mtp_h_input");
        *h_out = inp->h;
    }
    ggml_tensor * cur = inp->embd;
    res->add_input(std::move(inp));
    return cur;
}

class llm_graph_input_qwen4exp_ple : public llm_graph_input_i {
public:
    llm_graph_input_qwen4exp_ple(const llama_model & model,
                        const llama_kv_cache_context * mctx) : model(model), mctx(mctx) {}
    virtual ~llm_graph_input_qwen4exp_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        if (emb_host) { // [TAG_FN_PLE_HOST_GATHER]
            return emb_host->ne[1] == (int64_t) params.ubatch.n_tokens;
        }
        return rows->ne[0] == (int64_t) model.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    // [TAG_FN_PLE_HOST_GATHER] with host gather: F32 [ple_head_dim * ple_n_heads, n_tokens] instead of rows
    ggml_tensor * emb_host = nullptr;
    std::vector<float> emb_buf;
    std::vector<uint8_t> dio_raw; // [TAG_FN_PLE_DIRECT_IO] raw rows from the file

    const llama_model & model;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_qwen4exp_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hparams = model.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hparams.ple_image_token_id != 0
        ? (llama_token) hparams.ple_image_token_id
        : (llama_token) hparams.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hparams.ple_ngram_size;
    const int64_t n_heads  = hparams.ple_n_heads;
    const int64_t per_gram = hparams.ple_heads_per_ngram;
    const int64_t eos      = hparams.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hparams.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hparams.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hparams.ple_head_vocab_sizes[h_i] + hparams.ple_head_offsets[h_i]);
            }
        }
    }

    // [TAG_FN_PLE_DIRECT_IO] with direct I/O the rows come from the file: the mapping is never read, so not prefetched
    const auto & ple_dio = static_cast<const llama_model_qwen4exp &>(model).ple_dio;

    if (!ple_dio) {
        ggml_tensor * ple = model.per_layer_tok_embd;

        const bool prefetch = model.can_prefetch.count(ple);
        if (prefetch) {
            llama_prefetch_rows(ple, idx.data(), idx.size());
        }
    }

    if (emb_host) { // [TAG_FN_PLE_HOST_GATHER] same rows, dequantized here instead of by a CPU GET_ROWS (after the prefetch above)
        emb_buf.resize(ggml_nelements(emb_host));
        GGML_ASSERT((int64_t) emb_buf.size() == model.per_layer_tok_embd->ne[0]*n_heads*n_tokens);
        if (ple_dio) { // [TAG_FN_PLE_DIRECT_IO] the same bytes, read from the file instead of the mapping
            qwen4exp_rows_to_float_dio(*ple_dio, model.per_layer_tok_embd, idx.data(), n_heads*n_tokens, n_tokens,
                    dio_raw, emb_buf.data());
        } else {
            qwen4exp_rows_to_float(model.per_layer_tok_embd, idx.data(), n_heads*n_tokens, emb_buf.data());
        }
        ggml_backend_tensor_set(emb_host, emb_buf.data(), 0, emb_buf.size()*sizeof(float));
        return;
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // [TAG_FN_L3_GPU_CONVWB] copy the strided tail straight into the row: the slots' copies differ only by constant
        // pointer steps, so CUDA chains them into one launch ([TAG_CPY_CHAIN_FUSION]) instead of a 2-D memcpy and a
        // memcpy per slot (8 copy-engine nodes, ~28 us per GDN layer). Same bytes.
        ggml_tensor * cpy = ggml_cpy(ctx0, l3.convwb ? tail : ggml_cont(ctx0, tail), dst);

        // [TAG_FN_L3_GPU_DEFER] the next step reads these rows, this graph does not: after the bridge post
        if (l3.defer) {
            l3_deferred.push_back(cpy);
        } else {
            ggml_build_forward_expand(gf, cpy);
        }
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_qwen4exp_ple>(model, mctx_hyb->get_attn());

    // [TAG_FN_PLE_HOST_GATHER] the table rows arrive as an f32 input: no CPU GET_ROWS split
    // [TAG_FN_PLE_DIRECT_IO] always so with direct I/O: a GET_ROWS would read the rows through the mapping
    if (static_cast<const llama_model_qwen4exp &>(model).ple_dio || qwen4exp_host_gather_ok(model.per_layer_tok_embd)) {
        ple_inp->emb_host = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb_host);
        ggml_tensor * emb = ple_inp->emb_host;
        res->add_input(std::move(ple_inp));
        cb(emb, "ple_embd", -1);
        return emb;
    }

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
