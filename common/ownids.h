#pragma once

// [TAG_FN_R41_OWNIDS] round 41 "tok": a slot keeps the ids it generated when the next prompt brings its answer back.
//
// Why: the model can generate other ids than a fresh tokenization gives for the same bytes (a non-canonical split, for
// example ' p' '==' "='" "/'" where the tokenizer gives ' p' "==='" "/'", or '<parameter' '=' 'path' where it gives
// '<parameter' '=path'). A client that sends the answer back byte for byte then gets a prompt whose ids part from the
// slot's inside the answer, and with recurrent layers the slot can only go back to a checkpoint and read everything
// after it again (E:/r40/kyro-light/be1-replay.md: 6,991, 12,471 and 9,961 tokens in one agent run).
//
// What: the token match of the new prompt N against a slot's tokens S stops at k. From there both are walked by bytes
// (each token's piece with special tokens rendered, as the template writes them): while the bytes agree, every place
// where both have a token edge at the same byte is a sync point. The stretch between two sync points where the ids
// differ is replaced in N by S's ids. The walk stops where the bytes differ, where either side has a special token
// (control or user-defined: <|im_end|>, <think>, <tool_call>, ...) that the other side does not have at the same byte,
// at a media chunk the other side does not have at the same place, or at the end of either side. N keeps its own ids
// from the last sync point on, so its tail is the tokenizer's own tokenization in its context.
//
// Exact: the result has the same bytes as N (the replaced stretches have the same pieces), the same special tokens at
// the same bytes, the same media chunks in the same order, and equals S up to the last sync point. Header only, llama.h
// only.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// S[s0, s1) takes the place of N[n0, n1); both are text tokens only
struct common_ownids_run {
    size_t s0 = 0, s1 = 0;
    size_t n0 = 0, n1 = 0;
};

enum common_ownids_stop {
    COMMON_OWNIDS_END_S   = 0, // S is used up: the whole slot is matched
    COMMON_OWNIDS_END_N   = 1, // N is used up
    COMMON_OWNIDS_BYTES   = 2, // the bytes differ
    COMMON_OWNIDS_SPECIAL = 3, // a special token on one side only (or a different one)
    COMMON_OWNIDS_MEDIA   = 4, // a media chunk on one side only (or a different one)
};

struct common_ownids_result {
    size_t k     = 0; // the token match the walk started from
    size_t i_b   = 0; // S tokens matched after the walk (ids, then bytes)
    size_t j_b   = 0; // N tokens that cover the same bytes
    size_t bytes = 0; // bytes walked after k up to the last sync point
    int    stop  = COMMON_OWNIDS_END_S;
    std::vector<common_ownids_run> runs;

    size_t n_from_s() const { // ids taken from S
        size_t n = 0;
        for (const auto & r : runs) {
            n += r.s1 - r.s0;
        }
        return n;
    }
    size_t n_from_n() const { // ids of N they replace
        size_t n = 0;
        for (const auto & r : runs) {
            n += r.n1 - r.n0;
        }
        return n;
    }
};

// per vocabulary: every token's piece (special tokens rendered) and whether the tokenizer treats it as a special token
struct common_ownids_vocab {
    const llama_vocab *   vocab   = nullptr;
    int32_t               n_vocab = 0;
    std::string           pieces;
    std::vector<uint32_t> off;     // n_vocab + 1
    std::vector<uint8_t>  special; // CONTROL | USER_DEFINED | UNKNOWN

    bool ok() const { return n_vocab > 0 && off.size() == (size_t) n_vocab + 1; }
    bool valid(llama_token t) const { return t >= 0 && t < n_vocab; }
    const char * piece(llama_token t) const { return pieces.data() + off[t]; }
    size_t       len(llama_token t)   const { return off[t + 1] - off[t]; }
};

inline std::shared_ptr<const common_ownids_vocab> common_ownids_vocab_make(const llama_vocab * vocab) {
    auto v = std::make_shared<common_ownids_vocab>();
    v->vocab = vocab;
    const int32_t n = vocab ? llama_vocab_n_tokens(vocab) : 0;
    if (n <= 0) {
        return v;
    }
    v->off.resize((size_t) n + 1);
    v->special.resize((size_t) n);
    std::vector<char> buf(256);
    for (int32_t id = 0; id < n; ++id) {
        v->off[id] = (uint32_t) v->pieces.size();
        int32_t k = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
        if (k < 0) {
            buf.resize((size_t) -k);
            k = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
        }
        if (k > 0) {
            v->pieces.append(buf.data(), (size_t) k);
        }
        const auto attr = llama_vocab_get_attr(vocab, id);
        v->special[id] = (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) ? 1 : 0;
    }
    v->off[n] = (uint32_t) v->pieces.size();
    v->n_vocab = n;
    return v;
}

// the table of the last vocabulary asked for (one model per server); a new vocabulary address builds a new one
inline std::shared_ptr<const common_ownids_vocab> common_ownids_vocab_get(const llama_vocab * vocab) {
    static std::mutex mu;
    static std::shared_ptr<const common_ownids_vocab> last;
    std::lock_guard<std::mutex> lk(mu);
    if (!last || last->vocab != vocab || last->n_vocab != (vocab ? llama_vocab_n_tokens(vocab) : 0)) {
        last = common_ownids_vocab_make(vocab);
    }
    return last;
}

// the walk. S and N hold LLAMA_TOKEN_NULL where a media chunk sits. same_media(i, j) is called with S[i] and N[j] both at
// the start of a media chunk: the chunk's number of tokens when both are the same chunk, else 0. k <= min(nS, nN) is the
// token match (S[0, k) == N[0, k), media by id), from where the walk starts.
template <typename SameMedia>
inline common_ownids_result common_ownids_walk(const common_ownids_vocab & V, const llama_token * S, size_t nS,
        const llama_token * N, size_t nN, size_t k, SameMedia same_media) {
    common_ownids_result r;
    k = std::min(k, std::min(nS, nN));
    r.k = r.i_b = r.j_b = k;
    if (!V.ok()) {
        r.stop = COMMON_OWNIDS_SPECIAL;
        return r;
    }
    size_t i = k, j = k;   // the next token of each side
    size_t oi = 0, oj = 0; // bytes of S[i] / N[j] already compared (one of them is 0)
    size_t bytes = 0;      // bytes walked since k
    bool   diff = false;   // the stretch since the last sync point has other ids on the two sides
    size_t ds = 0, dn = 0; // where that stretch started
    for (;;) {
        if (oi == 0 && oj == 0) {
            // a sync point: both sides have a token edge at the same byte
            if (diff) {
                r.runs.push_back({ ds, i, dn, j });
                diff = false;
            }
            r.i_b   = i;
            r.j_b   = j;
            r.bytes = bytes;
            if (i == nS) {
                r.stop = COMMON_OWNIDS_END_S;
                break;
            }
            if (j == nN) {
                r.stop = COMMON_OWNIDS_END_N;
                break;
            }
            const llama_token a = S[i];
            const llama_token b = N[j];
            if (a == LLAMA_TOKEN_NULL || b == LLAMA_TOKEN_NULL) {
                const size_t n = (a == LLAMA_TOKEN_NULL && b == LLAMA_TOKEN_NULL) ? (size_t) same_media(i, j) : 0;
                if (n == 0 || i + n > nS || j + n > nN) {
                    r.stop = COMMON_OWNIDS_MEDIA;
                    break;
                }
                i += n;
                j += n;
                continue;
            }
            if (!V.valid(a) || !V.valid(b)) {
                r.stop = COMMON_OWNIDS_SPECIAL;
                break;
            }
            if (a == b) {
                bytes += V.len(a);
                ++i;
                ++j;
                continue;
            }
            if (V.special[a] || V.special[b]) {
                r.stop = COMMON_OWNIDS_SPECIAL;
                break;
            }
            diff = true;
            ds   = i;
            dn   = j;
        }
        // inside a stretch where the ids differ: both current tokens must be plain text with the same bytes
        if (i == nS || j == nN) {
            r.stop = i == nS ? COMMON_OWNIDS_END_S : COMMON_OWNIDS_END_N; // one side ends inside the other's token
            break;
        }
        const llama_token a = S[i];
        const llama_token b = N[j];
        if (a == LLAMA_TOKEN_NULL || b == LLAMA_TOKEN_NULL) {
            r.stop = COMMON_OWNIDS_MEDIA;
            break;
        }
        if (!V.valid(a) || !V.valid(b) || V.special[a] || V.special[b]) {
            r.stop = COMMON_OWNIDS_SPECIAL;
            break;
        }
        const size_t la = V.len(a) - oi;
        const size_t lb = V.len(b) - oj;
        const size_t m  = std::min(la, lb);
        if (m > 0 && std::memcmp(V.piece(a) + oi, V.piece(b) + oj, m) != 0) {
            r.stop = COMMON_OWNIDS_BYTES;
            break;
        }
        oi += m;
        oj += m;
        bytes += m;
        if (oi == V.len(a)) {
            ++i;
            oi = 0;
        }
        if (oj == V.len(b)) {
            ++j;
            oj = 0;
        }
    }
    return r;
}

// plain token lists (no media)
inline common_ownids_result common_ownids_walk(const common_ownids_vocab & V, const std::vector<llama_token> & S,
        const std::vector<llama_token> & N) {
    size_t k = 0;
    const size_t n = std::min(S.size(), N.size());
    while (k < n && S[k] == N[k] && S[k] != LLAMA_TOKEN_NULL) {
        ++k;
    }
    return common_ownids_walk(V, S.data(), S.size(), N.data(), N.size(), k, [](size_t, size_t) { return (size_t) 0; });
}

// N with the walk's runs put in (plain token lists)
inline std::vector<llama_token> common_ownids_apply(const std::vector<llama_token> & S, const std::vector<llama_token> & N,
        const std::vector<common_ownids_run> & runs) {
    std::vector<llama_token> out;
    out.reserve(N.size() + 64);
    size_t p = 0;
    for (const auto & r : runs) {
        out.insert(out.end(), N.begin() + p, N.begin() + r.n0);
        out.insert(out.end(), S.begin() + r.s0, S.begin() + r.s1);
        p = r.n1;
    }
    out.insert(out.end(), N.begin() + p, N.end());
    return out;
}
