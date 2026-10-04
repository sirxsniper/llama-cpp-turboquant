#include "llama-moetrace.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-graph.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// [TAG_FN_MOE_TRACE]

namespace {

constexpr int64_t DECODE_MAX_T = 8; // ubatches up to this many tokens count as decode (verify, multi-stream)

struct layer_counts {
    std::vector<uint64_t> decode_union;   // per decode ubatch: 1 for every expert used by any of its tokens
    std::vector<uint64_t> decode_tokens;  // per decode token and routed slot
    std::vector<uint64_t> prefill_tokens; // per prefill token and routed slot
    std::vector<double>   decode_weight;  // sum of routing weights over decode tokens
};

struct trace_state {
    std::mutex mtx;

    std::string profile_path;
    std::string trace_path;
    std::string source;
    int64_t     every     = 256;
    int64_t     max_bytes = 4096ll << 20;

    FILE *   ftrace       = nullptr;
    int64_t  trace_bytes  = 0;
    bool     trace_capped = false;

    int64_t n_expert = 0;
    std::map<int, layer_counts> layers;

    uint64_t n_ubatch         = 0;
    uint64_t n_steps_decode   = 0;
    uint64_t n_tokens_decode  = 0;
    uint64_t n_tokens_prefill = 0;
    uint64_t n_ubatch_draft   = 0;

    ~trace_state() {
        write_profile();
        if (ftrace) {
            fclose(ftrace);
            ftrace = nullptr;
        }
    }

    void write_profile() {
        if (profile_path.empty() || layers.empty()) {
            return;
        }
        const std::string tmp = profile_path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "moe-trace: cannot write %s\n", tmp.c_str()); // may run during static destruction
            return;
        }
        const int n_layer = layers.rbegin()->first + 1;
        fprintf(f, "moeprof v1 n_layer=%d n_expert=%lld steps=%llu decode_tokens=%llu prefill_tokens=%llu draft_ubatches=%llu source=%s\n",
                n_layer, (long long) n_expert, (unsigned long long) n_steps_decode, (unsigned long long) n_tokens_decode,
                (unsigned long long) n_tokens_prefill, (unsigned long long) n_ubatch_draft, source.empty() ? "-" : source.c_str());
        auto dump = [&](const char * key, auto get) {
            for (const auto & [il, lc] : layers) {
                fprintf(f, "%s %d", key, il);
                for (int64_t e = 0; e < n_expert; ++e) {
                    fprintf(f, " %llu", (unsigned long long) get(lc, e));
                }
                fputc('\n', f);
            }
        };
        dump("decode_union",   [](const layer_counts & lc, int64_t e) { return lc.decode_union[e]; });
        dump("decode_tokens",  [](const layer_counts & lc, int64_t e) { return lc.decode_tokens[e]; });
        dump("prefill_tokens", [](const layer_counts & lc, int64_t e) { return lc.prefill_tokens[e]; });
        // weights as integer micro-units, so every section parses the same way
        dump("decode_weight_u", [](const layer_counts & lc, int64_t e) { return (uint64_t) (lc.decode_weight[e]*1e6); });
        fclose(f);
        std::error_code ec;
        std::filesystem::rename(tmp, profile_path, ec);
        if (ec) {
            std::filesystem::remove(profile_path, ec);
            std::filesystem::rename(tmp, profile_path, ec);
        }
    }
};

trace_state * g_state() {
    static trace_state st;
    return &st;
}

struct env_cfg {
    bool active = false;
    int  pred_k = 0;
};

const env_cfg & cfg() {
    static const env_cfg c = [] {
        env_cfg r;
        const char * p = getenv("LLAMA_MOE_PROFILE");
        const char * t = getenv("LLAMA_MOE_TRACE");
        const char * k = getenv("LLAMA_MOE_TRACE_PRED");
        r.active = (p && *p) || (t && *t);
        if (k && atoi(k) != 0) {
            const char * kk = getenv("LLAMA_MOE_TRACE_PRED_K");
            r.pred_k = kk ? std::max(1, std::min(64, atoi(kk))) : 16;
        }
        if (r.active) {
            trace_state * st = g_state();
            st->profile_path = p ? p : "";
            st->trace_path   = t ? t : "";
            if (const char * s = getenv("LLAMA_MOE_TRACE_SOURCE")) {
                st->source = s;
                for (char & ch : st->source) {
                    if (ch == '\n' || ch == '\r') { ch = ' '; }
                }
            }
            if (const char * e = getenv("LLAMA_MOE_PROFILE_EVERY")) {
                st->every = std::max(1, atoi(e));
            }
            if (const char * m = getenv("LLAMA_MOE_TRACE_MAX_MB")) {
                st->max_bytes = (int64_t) std::max(1, atoi(m)) << 20;
            }
            LLAMA_LOG_INFO("moe-trace: profile=%s trace=%s pred_k=%d\n", st->profile_path.empty() ? "-" : st->profile_path.c_str(),
                    st->trace_path.empty() ? "-" : st->trace_path.c_str(), r.pred_k);
        }
        return r;
    }();
    return c;
}

void read_tensor(ggml_backend_sched * sched, const ggml_tensor * t, void * dst) {
    ggml_tensor * tt = const_cast<ggml_tensor *>(t);
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, tt);
    if (b) {
        ggml_backend_tensor_get_async(b, tt, dst, 0, ggml_nbytes(tt));
    } else {
        ggml_backend_tensor_get(tt, dst, 0, ggml_nbytes(tt));
    }
}

} // namespace

bool llama_moe_trace_active() {
    return cfg().active;
}

int llama_moe_trace_pred_k() {
    return cfg().active ? cfg().pred_k : 0;
}

void llama_moe_trace_flush() {
    if (!cfg().active) {
        return;
    }
    trace_state * st = g_state();
    std::lock_guard<std::mutex> lk(st->mtx);
    st->write_profile();
    if (st->ftrace) {
        fflush(st->ftrace);
    }
}

void llama_moe_trace_collect(ggml_backend_sched * sched, const llm_graph_result * res, const llama_ubatch & ubatch, bool is_draft) {
    if (!cfg().active || res->t_moe_ids.empty()) {
        return;
    }
    const size_t n_l = res->t_moe_ids.size();

    thread_local std::vector<std::vector<int32_t>> ids;
    thread_local std::vector<std::vector<float>>   w;
    thread_local std::vector<std::vector<int32_t>> pred;
    ids.resize(n_l);
    w.resize(n_l);
    pred.resize(n_l);
    for (size_t i = 0; i < n_l; ++i) {
        const ggml_tensor * ti = res->t_moe_ids[i];
        const ggml_tensor * tw = res->t_moe_w[i];
        const ggml_tensor * tp = res->t_moe_pred[i];
        ids[i].resize(ggml_nelements(ti));
        read_tensor(sched, ti, ids[i].data());
        w[i].resize(tw ? ggml_nelements(tw) : 0);
        if (tw) {
            read_tensor(sched, tw, w[i].data());
        }
        pred[i].resize(tp ? ggml_nelements(tp) : 0);
        if (tp) {
            read_tensor(sched, tp, pred[i].data());
        }
    }
    ggml_backend_sched_synchronize(sched);

    trace_state * st = g_state();
    std::lock_guard<std::mutex> lk(st->mtx);

    const int64_t T_ub    = ubatch.n_tokens;
    const bool    prefill = T_ub > DECODE_MAX_T;
    st->n_ubatch++;
    if (is_draft) {
        st->n_ubatch_draft++;
    } else if (prefill) {
        st->n_tokens_prefill += T_ub;
    } else {
        st->n_steps_decode++;
        st->n_tokens_decode += T_ub;
    }

    std::vector<uint8_t> seen;
    for (size_t i = 0; i < n_l; ++i) {
        const int il = res->t_moe_il[i];
        const ggml_tensor * ti = res->t_moe_ids[i];
        const int64_t n_used = ti->ne[0];
        const int64_t T      = ti->ne[1];
        const int64_t n_exp  = res->t_moe_n_expert[i];
        if (st->n_expert == 0) {
            st->n_expert = n_exp;
        }
        if (n_exp != st->n_expert) {
            continue; // mixed expert counts are not profiled
        }
        auto & lc = st->layers[il];
        if (lc.decode_union.empty()) {
            lc.decode_union.assign(n_exp, 0);
            lc.decode_tokens.assign(n_exp, 0);
            lc.prefill_tokens.assign(n_exp, 0);
            lc.decode_weight.assign(n_exp, 0.0);
        }
        seen.assign(n_exp, 0);
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t k = 0; k < n_used; ++k) {
                const int32_t e = ids[i][t*n_used + k];
                if (e < 0 || e >= n_exp) {
                    continue;
                }
                if (prefill) {
                    lc.prefill_tokens[e]++;
                } else {
                    lc.decode_tokens[e]++;
                    if (!w[i].empty()) {
                        lc.decode_weight[e] += w[i][t*n_used + k];
                    }
                    if (!seen[e]) {
                        seen[e] = 1;
                        lc.decode_union[e]++;
                    }
                }
            }
        }
    }

    // binary trace record
    if (!st->trace_path.empty() && !st->trace_capped) {
        if (!st->ftrace) {
            st->ftrace = fopen(st->trace_path.c_str(), "wb");
            if (!st->ftrace) {
                LLAMA_LOG_WARN("moe-trace: cannot open %s\n", st->trace_path.c_str());
                st->trace_path.clear();
                return;
            }
            const char magic[4] = {'M', 'O', 'E', 'T'};
            const uint32_t ver = 1, n_exp = (uint32_t) st->n_expert;
            fwrite(magic, 1, 4, st->ftrace);
            fwrite(&ver, 4, 1, st->ftrace);
            fwrite(&n_exp, 4, 1, st->ftrace);
            st->trace_bytes = 12;
        }
        std::vector<uint8_t> rec;
        auto put = [&](const void * p, size_t n) { rec.insert(rec.end(), (const uint8_t *) p, (const uint8_t *) p + n); };
        const uint32_t step  = (uint32_t) st->n_ubatch;
        const uint16_t T16   = (uint16_t) T_ub;
        const uint8_t  flags = (uint8_t) ((prefill ? 1 : 0) | (is_draft ? 2 : 0) | (!prefill && T_ub > 1 ? 4 : 0));
        const uint8_t  nl    = (uint8_t) n_l;
        put(&step, 4); put(&T16, 2); put(&flags, 1); put(&nl, 1);
        for (int64_t t = 0; t < T_ub; ++t) {
            const uint16_t s = (ubatch.seq_id && ubatch.n_seq_id && ubatch.n_seq_id[t] > 0) ? (uint16_t) ubatch.seq_id[t][0] : 0;
            put(&s, 2);
        }
        for (size_t i = 0; i < n_l; ++i) {
            const ggml_tensor * ti = res->t_moe_ids[i];
            const uint16_t il     = (uint16_t) res->t_moe_il[i];
            const uint16_t T      = (uint16_t) ti->ne[1];
            const uint8_t  n_used = (uint8_t) ti->ne[0];
            const ggml_tensor * tp = res->t_moe_pred[i];
            const uint8_t  pk     = tp ? (uint8_t) tp->ne[0] : 0;
            const uint16_t Tp     = tp ? (uint16_t) tp->ne[1] : 0;
            const uint16_t pad    = 0;
            put(&il, 2); put(&T, 2); put(&n_used, 1); put(&pk, 1); put(&Tp, 2); put(&pad, 2);
            for (int64_t j = 0; j < (int64_t) T*n_used; ++j) {
                const uint16_t v = (uint16_t) ids[i][j];
                put(&v, 2);
            }
            for (int64_t j = 0; j < (int64_t) T*n_used; ++j) {
                const ggml_fp16_t v = ggml_fp32_to_fp16(w[i].empty() ? 0.0f : w[i][j]);
                put(&v, 2);
            }
            for (int64_t j = 0; j < (int64_t) Tp*pk; ++j) {
                const uint16_t v = (uint16_t) pred[i][j];
                put(&v, 2);
            }
        }
        const uint32_t len = (uint32_t) rec.size();
        if (st->trace_bytes + 4 + (int64_t) len > st->max_bytes) {
            st->trace_capped = true;
            LLAMA_LOG_WARN("moe-trace: %s reached LLAMA_MOE_TRACE_MAX_MB, no more records\n", st->trace_path.c_str());
        } else {
            fwrite(&len, 4, 1, st->ftrace);
            fwrite(rec.data(), 1, rec.size(), st->ftrace);
            st->trace_bytes += 4 + len;
            if (st->n_ubatch % 64 == 0) {
                fflush(st->ftrace);
            }
        }
    }

    if (st->n_ubatch % st->every == 0) {
        st->write_profile();
    }
}
