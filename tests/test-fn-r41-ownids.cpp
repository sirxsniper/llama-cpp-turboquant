// [TAG_FN_R41_OWNIDS] common/ownids.h: a prompt that brings a slot's answer back byte for byte takes the slot's own ids.
// No model weights and no GPU: the tokenizer alone (a vocab-only GGUF, argv[1]; ctest passes models/ggml-vocab-qwen35.gguf,
// the 248K vocabulary of Qwen3.5 .. 3.8; the Flash-Next GGUF's first shard can be given instead). argv[2] (optional): a
// rendered prompt to draw more cases from (for example the owner's r2_rendered.txt; nothing of it is stored here).
//
// The slot's ids S stand for what the model generated: the prompt's own tokenization, then the answer as pieces the model
// may generate (the three splits of the owner's agent run of 2026-11-14 among them: ' p' '==' "='" "/'" where the tokenizer
// gives ' p' "==='" "/'", '<parameter' '=' 'path' where it gives '<parameter' '=path', '~' "/'+" where it gives '~/' "'+").
// The next prompt N is the tokenizer's ids of the prompt + the answer + the tool result. Checked for every case:
//   - the walk takes S's ids through the whole answer when the bytes agree (i_b == |S|, stop end-of-holder);
//   - the result has the bytes of N, the special tokens of N at the same bytes, S's ids up to i_b and N's own ids after;
//   - it stops at the first byte that differs, at a special token the other side lacks, at a media chunk that differs;
//   - a multi-byte character split across tokens, a run of byte tokens, the thinking and tool-call tags pass;
//   - random re-splits of real text (many thousand cases) keep all of the above, and the stop is the last common edge.

#include "llama.h"
#include "ownids.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int n_fail  = 0;
static int n_check = 0;

#define CHECK(cond, ...) do { n_check++; if (!(cond)) { n_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const llama_vocab * g_vocab = nullptr;
static std::shared_ptr<const common_ownids_vocab> g_V;

static std::vector<llama_token> tok(const std::string & s, bool special = true) {
    std::vector<llama_token> out(s.size() + 16);
    int n = llama_tokenize(g_vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(), false, special);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(g_vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(), false, special);
    }
    out.resize(std::max(n, 0));
    return out;
}

static std::string text_of(const std::vector<llama_token> & ids) {
    std::string s;
    for (const llama_token t : ids) {
        if (t == LLAMA_TOKEN_NULL) {
            s += '\0';
        } else {
            s.append(g_V->piece(t), g_V->len(t));
        }
    }
    return s;
}

// (byte offset, id) of every special token
static std::vector<std::pair<size_t, llama_token>> specials_of(const std::vector<llama_token> & ids) {
    std::vector<std::pair<size_t, llama_token>> out;
    size_t b = 0;
    for (const llama_token t : ids) {
        if (t == LLAMA_TOKEN_NULL) {
            b++;
            continue;
        }
        if (g_V->special[t]) {
            out.emplace_back(b, t);
        }
        b += g_V->len(t);
    }
    return out;
}

static std::string show(const std::vector<llama_token> & ids, size_t a, size_t n) {
    std::string out;
    for (size_t x = a; x < ids.size() && x < a + n; ++x) {
        out += "'";
        for (size_t q = 0; q < g_V->len(ids[x]); ++q) {
            const unsigned char c = (unsigned char) g_V->piece(ids[x])[q];
            char buf[8];
            if (c == '\n') {
                out += "\\n";
            } else if (c < 0x20 || c >= 0x7f) {
                snprintf(buf, sizeof(buf), "\\x%02x", c);
                out += buf;
            } else {
                out += (char) c;
            }
        }
        out += "' ";
    }
    return out;
}

// the invariants of a walk result and its applied list; returns the applied list
static std::vector<llama_token> check_result(const char * name, const std::vector<llama_token> & S, const std::vector<llama_token> & N,
        const common_ownids_result & r) {
    const std::vector<llama_token> M = common_ownids_apply(S, N, r.runs);
    CHECK(text_of(M) == text_of(N), "%s: the result has other bytes than the prompt", name);
    CHECK(specials_of(M) == specials_of(N), "%s: the special tokens moved", name);
    CHECK(M.size() >= r.i_b && std::equal(S.begin(), S.begin() + r.i_b, M.begin()), "%s: the result is not S up to i_b", name);
    const size_t tail = N.size() - r.j_b;
    CHECK(M.size() == r.i_b + tail && std::equal(N.begin() + r.j_b, N.end(), M.begin() + r.i_b), "%s: the tail is not N's own ids", name);
    for (size_t x = 1; x < r.runs.size(); ++x) {
        CHECK(r.runs[x].n0 >= r.runs[x - 1].n1 && r.runs[x].s0 >= r.runs[x - 1].s1, "%s: runs out of order", name);
    }
    return M;
}

// the walk stopped at the last place both sides have a token edge before the first byte that differs
static void check_maximal(const char * name, const std::vector<llama_token> & S, const std::vector<llama_token> & N,
        const common_ownids_result & r) {
    const std::string ts = text_of(S), tn = text_of(N);
    size_t d = 0;
    while (d < ts.size() && d < tn.size() && ts[d] == tn[d]) {
        d++;
    }
    std::vector<size_t> es{ 0 }, en{ 0 };
    for (const llama_token t : S) { es.push_back(es.back() + (t == LLAMA_TOKEN_NULL ? 1 : g_V->len(t))); }
    for (const llama_token t : N) { en.push_back(en.back() + (t == LLAMA_TOKEN_NULL ? 1 : g_V->len(t))); }
    const size_t b_sync = es[r.i_b];
    CHECK(b_sync == en[r.j_b], "%s: the sync point is not at the same byte (%zu, %zu)", name, b_sync, en[r.j_b]);
    CHECK(b_sync <= d, "%s: synced past the first differing byte (%zu > %zu)", name, b_sync, d);
    if (r.stop == COMMON_OWNIDS_BYTES || r.stop == COMMON_OWNIDS_END_N || r.stop == COMMON_OWNIDS_END_S) {
        // no common edge in (b_sync, d] (a common edge at d itself is a sync point the walk reaches)
        size_t common = 0;
        for (size_t x = r.i_b + 1; x < es.size(); ++x) {
            if (es[x] > d) {
                break;
            }
            if (es[x] > b_sync && std::binary_search(en.begin(), en.end(), es[x])) {
                common = es[x];
                break;
            }
        }
        CHECK(common == 0, "%s: the walk stopped at byte %zu but both sides have an edge at %zu (first difference %zu, stop %d)",
                name, b_sync, common, d, r.stop);
    }
}

// the byte token of every byte (byte-level BPE has one per byte)
static llama_token g_byte_id[256];

// ids with exactly the bytes of p: its tokenization, or (p is not whole UTF-8, the tokenizer would put U+FFFD there) one byte
// token per byte, as the model generates a character it splits
static std::vector<llama_token> ids_for(const std::string & p, bool special = true) {
    std::vector<llama_token> t = tok(p, special);
    if (text_of(t) == p) {
        return t;
    }
    t.clear();
    for (const unsigned char c : p) {
        t.push_back(g_byte_id[c]);
    }
    return t;
}

// a model-like answer: the pieces given, each tokenized alone (a piece the tokenizer splits stays split)
static std::vector<llama_token> pieces_ids(const std::vector<std::string> & pieces) {
    std::vector<llama_token> out;
    for (const auto & p : pieces) {
        const auto t = ids_for(p);
        out.insert(out.end(), t.begin(), t.end());
    }
    return out;
}

static std::string join(const std::vector<std::string> & pieces) {
    std::string s;
    for (const auto & p : pieces) {
        s += p;
    }
    return s;
}

static const std::string k_prompt =
    "<|im_start|>system\nYou are a coding assistant.<|im_end|>\n"
    "<|im_start|>user\nBuild a small browser desktop in one HTML file.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n";

static const std::string k_next =
    "<|im_end|>\n<|im_start|>user\n<tool_response>\nCreated nimbus-os.html (307 lines)\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\n";

// one scenario: S = tok(prompt) + the answer's pieces, N = tok(prompt + answer + next)
static common_ownids_result scenario(const char * name, const std::vector<std::string> & answer, bool expect_full,
        const std::string & next = k_next) {
    std::vector<llama_token> S = tok(k_prompt);
    const std::vector<llama_token> A = pieces_ids(answer);
    S.insert(S.end(), A.begin(), A.end());
    const std::vector<llama_token> N = tok(k_prompt + join(answer) + next);
    const common_ownids_result r = common_ownids_walk(*g_V, S, N);
    check_result(name, S, N, r);
    check_maximal(name, S, N, r);
    if (expect_full) {
        CHECK(r.i_b == S.size() && r.stop == COMMON_OWNIDS_END_S, "%s: the whole answer should be matched (i_b %zu of %zu, stop %d)",
                name, r.i_b, S.size(), r.stop);
    }
    printf("  %-44s S %5zu N %5zu  token match %5zu -> %5zu  runs %2zu  stop %d\n", name, S.size(), N.size(), r.k, r.i_b,
            r.runs.size(), r.stop);
    return r;
}

static void test_real_cuts() {
    printf("the splits of the owner's agent run (2026-11-14)\n");
    // the tokenizer must split these as the finding says, else the cases do not test what they claim
    {
        const auto a = tok(" p==='/'");
        printf("    tokenizer: ' p===\\'/\\'' -> %s\n", show(a, 0, 8).c_str());
        const auto b = tok("<parameter=path>");
        printf("    tokenizer: '<parameter=path>' -> %s\n", show(b, 0, 8).c_str());
        const auto c = tok("?'~/'+cwd");
        printf("    tokenizer: '?\\'~/\\'+cwd' -> %s\n", show(c, 0, 8).c_str());
    }
    // round 13.39: in the thinking, "parent = p==='/'"
    scenario("thinking: ' p' '==' \"='\" \"/'\"",
             { "The", " root", " check", ":", " const", " parent", " =", " p", "==", "='", "/'", " ?", " null", " :", " dir",
               "(p", ");", "\n</think>\n\n", "Next", " the", " shell", "." },
             true);
    // rounds 17.39, 21.54, 22.22 ...: '<parameter' then not the single token '=path'
    scenario("tool call: '<parameter' '=' 'path'",
             { "Writing", " the", " file", ".", "\n</think>\n\n", "<tool_call>", "\n<function=edit_file>\n", "<parameter", "=",
               "path", ">\n", "nimbus", "-os", ".html", "\n</parameter>\n", "<parameter", "=", "old_string", ">\n\n</parameter>\n",
               "</function>\n", "</tool_call>" },
             true);
    // round 19.52: in the thinking, "?'~/'+cwd"
    scenario("thinking: '~' \"/'+\"",
             { "Path", " shown", " as", " h", "?'", "~", "/'+", "cwd", ":", "cwd", ".", "\n</think>\n\n", "Done", "." },
             true);
}

static void test_shapes() {
    printf("shapes\n");
    // a multi-byte character split across two tokens, and a run of single-byte tokens
    scenario("utf-8: 'caf' '\\xc3' '\\xa9' split",
             { "The", " caf", "\xc3", "\xa9", " menu", " \xe2\x9c", "\x93", " done", "\n</think>\n\n", "OK" }, true);
    scenario("utf-8: emoji as four byte tokens",
             { "Icon", ": ", "\xf0", "\x9f", "\x96", "\xa5", " ok", "\n</think>\n\n", "Fine" }, true);
    // the newline the stream holds back before </think>: '\n' then '\nparent'
    scenario("newlines: '\\n' '\\nparent'", { "Plan", ".", "\n", "\nparent", " first", "\n</think>\n\n", "Go" }, true);
    // thinking and tool-call tags as the special tokens the model generates
    scenario("tags: </think> <tool_call> </tool_call>",
             { "x", "\n", "</think>", "\n\n", "<tool_call>", "\n<function=todo_write>\n<parameter=items>\n[{\"id\": \"1\"}]\n</parameter>\n</function>\n",
               "</tool_call>" }, true);
    // the answer ends with an answer token that the next prompt's <|im_end|> follows directly: no edge problem
    scenario("end: last token right before <|im_end|>", { "Done", " now", "!" }, true);
    // canonical answer: nothing to change, the token match alone covers it
    {
        std::vector<llama_token> S = tok(k_prompt + "All good.\n</think>\n\nYes.");
        std::vector<llama_token> N = tok(k_prompt + "All good.\n</think>\n\nYes." + k_next);
        const auto r = common_ownids_walk(*g_V, S, N);
        CHECK(r.runs.empty() && r.i_b == S.size() && r.k == S.size(), "canonical answer: no runs expected");
        check_result("canonical", S, N, r);
    }
}

static void test_stops() {
    printf("stops\n");
    // the bytes differ inside the answer (the client trimmed a space): S is taken up to the last common edge before it
    {
        std::vector<std::string> ans = { "Use", " p", "==", "='", "/'", " here", ".", "  ", "\n</think>\n\n", "Ok" };
        std::vector<llama_token> S = tok(k_prompt);
        const auto A = pieces_ids(ans);
        S.insert(S.end(), A.begin(), A.end());
        const std::string replay = "Use p==='/' here.\n</think>\n\nOk"; // the two spaces dropped
        const std::vector<llama_token> N = tok(k_prompt + replay + k_next);
        const auto r = common_ownids_walk(*g_V, S, N);
        check_result("bytes differ", S, N, r);
        check_maximal("bytes differ", S, N, r);
        CHECK(r.stop == COMMON_OWNIDS_BYTES && r.i_b < S.size() && r.i_b > r.k && !r.runs.empty(),
                "bytes differ: expected a partial match with runs, stop bytes (i_b %zu of %zu, k %zu, stop %d)", r.i_b, S.size(), r.k, r.stop);
        printf("  %-44s token match %zu -> %zu of %zu, stop %d\n", "bytes differ after the split", r.k, r.i_b, S.size(), r.stop);
    }
    // the model wrote a tag as plain pieces where the next prompt has the special token: the walk stops right before it.
    // Whether a tag is a special token depends on the vocabulary: in the Flash-Next GGUF <tool_call> is user-defined (type 4)
    // and <|im_start|> is control (type 3); the 151,936-token models/ggml-vocab-qwen35.gguf has no <tool_call> token at all
    // (both sides then hold the same plain bytes there and the walk rightly goes through). So the expectation follows the
    // vocabulary: a stop when the tag tokenizes to one special token, else the whole slot.
    for (const auto & tag : std::vector<std::vector<std::string>>{ { "<", "tool", "_call", ">" }, { "<", "|", "im", "_start", "|", ">" } }) {
        const std::string text = join(tag);
        const auto one = tok(text);
        const bool is_special = one.size() == 1 && g_V->special[one[0]];
        std::vector<llama_token> S = tok(k_prompt);
        const auto A1 = pieces_ids({ "Call", " p", "==", "='", "\n</think>\n\n" });
        // plain text pieces, each tokenized alone (tokenizing the whole tag with parse_special off would still give a
        // user-defined token: llama.cpp skips only control and unknown tokens then)
        const auto A2 = pieces_ids(tag);
        const auto A3 = tok("\n<function=x>\n</function>\n");
        S.insert(S.end(), A1.begin(), A1.end());
        const size_t at = S.size();
        S.insert(S.end(), A2.begin(), A2.end());
        S.insert(S.end(), A3.begin(), A3.end());
        const std::vector<llama_token> N = tok(k_prompt + "Call p==='\n</think>\n\n" + text + "\n<function=x>\n</function>\n" + k_next);
        const auto r = common_ownids_walk(*g_V, S, N);
        const std::string name = "plain " + text;
        check_result(name.c_str(), S, N, r);
        check_maximal(name.c_str(), S, N, r);
        if (is_special) {
            CHECK(r.stop == COMMON_OWNIDS_SPECIAL && r.i_b == at, "%s: expected the stop right before it (i_b %zu, at %zu, stop %d)",
                    name.c_str(), r.i_b, at, r.stop);
        } else {
            CHECK(r.stop == COMMON_OWNIDS_END_S && r.i_b == S.size(), "%s: not a special token here, the whole slot expected (i_b %zu of %zu, stop %d)",
                    name.c_str(), r.i_b, S.size(), r.stop);
        }
        printf("  %-44s %s: token match %zu -> %zu (tag at %zu), stop %d\n", name.c_str(),
                is_special ? "special token in this vocabulary" : "NOT a token in this vocabulary", r.k, r.i_b, at, r.stop);
    }
    // media: the same chunk on both sides passes, another chunk stops the walk there
    {
        std::vector<llama_token> base = tok(k_prompt);
        std::vector<llama_token> S = base, N = base;
        const auto t1 = pieces_ids({ "see", " p", "==", "='" });
        const auto t1n = tok("see p==='");
        S.insert(S.end(), t1.begin(), t1.end());
        N.insert(N.end(), t1n.begin(), t1n.end());
        const size_t ms = S.size(), mn = N.size();
        for (int x = 0; x < 5; ++x) { S.push_back(LLAMA_TOKEN_NULL); N.push_back(LLAMA_TOKEN_NULL); }
        const auto t2 = pieces_ids({ " after", " ~", "/'+", "x" });
        const auto t2n = tok(" after ~/'+x");
        S.insert(S.end(), t2.begin(), t2.end());
        N.insert(N.end(), t2n.begin(), t2n.end());
        const auto tn = tok(k_next);
        N.insert(N.end(), tn.begin(), tn.end());
        size_t k = 0;
        while (k < S.size() && k < N.size() && S[k] == N[k]) { k++; }
        for (int same = 1; same >= 0; --same) {
            const auto r = common_ownids_walk(*g_V, S.data(), S.size(), N.data(), N.size(), k, [&](size_t i, size_t j) -> size_t {
                return (i == ms && j == mn && same) ? 5 : 0;
            });
            if (same) {
                const size_t want_runs = (t1 != t1n ? 1 : 0) + (t2 != t2n ? 1 : 0);
                CHECK(want_runs >= 1 && t1 != t1n, "media: the stretch before the chunk should be split another way than the tokenizer's");
                CHECK(r.i_b == S.size() && r.runs.size() == want_runs, "media: the same chunk should pass (i_b %zu of %zu, runs %zu)", r.i_b, S.size(), r.runs.size());
                const auto M = common_ownids_apply(S, N, r.runs);
                CHECK(std::equal(S.begin(), S.end(), M.begin()), "media: result not S");
                CHECK(std::count(M.begin(), M.end(), LLAMA_TOKEN_NULL) == 5, "media: chunk lost");
            } else {
                CHECK(r.stop == COMMON_OWNIDS_MEDIA && r.i_b == ms, "media: another chunk should stop at it (i_b %zu, at %zu)", r.i_b, ms);
            }
            printf("  %-44s token match %zu -> %zu of %zu, stop %d\n", same ? "media: same chunk" : "media: another chunk", r.k, r.i_b, S.size(), r.stop);
        }
    }
    // the prompt ends inside the slot's tokens (a shorter replay): S is taken as far as N goes
    {
        std::vector<llama_token> S = tok(k_prompt);
        const auto A = pieces_ids({ "a", " p", "==", "='", "/'", " b", " c", " d" });
        S.insert(S.end(), A.begin(), A.end());
        const std::vector<llama_token> N = tok(k_prompt + "a p==='/' b");
        const auto r = common_ownids_walk(*g_V, S, N);
        check_result("prompt ends inside", S, N, r);
        CHECK(r.stop == COMMON_OWNIDS_END_N && r.j_b == N.size(), "prompt ends inside: expected end-of-prompt (stop %d, j_b %zu of %zu)", r.stop, r.j_b, N.size());
    }
}

// random re-splits of text the model could have written: every token of the answer may be cut at a random byte into two
// pieces tokenized alone; the next prompt is the tokenizer's own ids. Bytes changed at random in some cases.
static void test_fuzz(const std::string & corpus, int n_cases, unsigned seed) {
    printf("fuzz: %d cases from %zu bytes of text\n", n_cases, corpus.size());
    std::mt19937 rng(seed);
    int n_full = 0, n_partial = 0, n_runs = 0;
    size_t ids_from_s = 0;
    for (int c = 0; c < n_cases; ++c) {
        // a window of the corpus as the answer, cut at whitespace so it starts a word
        const size_t len = 200 + rng() % 6000;
        size_t a = rng() % (corpus.size() > len ? corpus.size() - len : 1);
        while (a < corpus.size() && a > 0 && corpus[a - 1] != ' ' && corpus[a - 1] != '\n') { a++; }
        if (a >= corpus.size()) {
            continue;
        }
        std::string answer = corpus.substr(a, len);
        // the prompt part: the answer must not start inside a special token's text
        const std::vector<llama_token> canon = tok(answer);
        std::vector<llama_token> A;
        const int cut_rate = 1 + (int) (rng() % 20); // percent of tokens cut
        for (const llama_token t : canon) {
            const size_t L = g_V->len(t);
            if (!g_V->special[t] && L >= 2 && (int) (rng() % 100) < cut_rate) {
                const size_t at = 1 + rng() % (L - 1);
                const std::string p(g_V->piece(t), L);
                const auto x = ids_for(p.substr(0, at), false);
                const auto y = ids_for(p.substr(at), false);
                A.insert(A.end(), x.begin(), x.end());
                A.insert(A.end(), y.begin(), y.end());
            } else {
                A.push_back(t);
            }
        }
        std::vector<llama_token> S = tok(k_prompt);
        S.insert(S.end(), A.begin(), A.end());
        std::string replay = answer;
        const bool change = rng() % 4 == 0;
        if (change && replay.size() > 10) {
            replay[rng() % replay.size()] ^= 0x01; // one byte differs
        }
        const std::vector<llama_token> N = tok(k_prompt + replay + k_next);
        const auto r = common_ownids_walk(*g_V, S, N);
        char name[64];
        snprintf(name, sizeof(name), "fuzz case %d", c);
        check_result(name, S, N, r);
        check_maximal(name, S, N, r);
        // with the bytes unchanged the whole answer is matched (S's text is a prefix of N's text and ends at a token edge of N:
        // <|im_end|> follows), unless the answer itself holds special-token text the slot has as plain pieces
        if (!change) {
            // the cuts never split a special token, so nothing stops the walk before the end of S
            {
                CHECK(r.i_b == S.size() && r.stop == COMMON_OWNIDS_END_S, "%s: unchanged bytes but i_b %zu of %zu (stop %d)", name, r.i_b, S.size(), r.stop);
            }
            n_full += r.i_b == S.size() ? 1 : 0;
        } else {
            n_partial++;
        }
        n_runs += (int) r.runs.size();
        ids_from_s += r.n_from_s();
    }
    printf("  unchanged bytes matched whole: %d, changed: %d, runs %d, ids taken from the slot %zu\n", n_full, n_partial, n_runs, ids_from_s);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "models/ggml-vocab-qwen35.gguf";
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        printf("FAIL: cannot load the vocabulary from %s\n", path);
        return 1;
    }
    g_vocab = llama_model_get_vocab(model);
    g_V     = common_ownids_vocab_get(g_vocab);
    printf("vocabulary %s: %d tokens, piece table %zu bytes\n", path, g_V->n_vocab, g_V->pieces.size());
    CHECK(g_V->ok(), "no piece table");
    {
        int found = 0;
        for (int c = 0; c < 256; ++c) {
            g_byte_id[c] = LLAMA_TOKEN_NULL;
        }
        for (llama_token t = 0; t < g_V->n_vocab; ++t) {
            if (g_V->len(t) == 1 && !g_V->special[t] && g_byte_id[(unsigned char) g_V->piece(t)[0]] == LLAMA_TOKEN_NULL) {
                g_byte_id[(unsigned char) g_V->piece(t)[0]] = t;
                found++;
            }
        }
        CHECK(found == 256, "byte tokens: %d of 256", found);
    }

    test_real_cuts();
    test_shapes();
    test_stops();

    // fuzz: a synthetic corpus (code, prose, JSON, multi-byte text, tags), then the given file
    std::string synth;
    {
        const char * parts[] = {
            "function open(p){ const parent = p==='/' ? null : dir(p); return parent; }\n",
            "The window manager keeps a z-order list; focusing a window moves it to the end. ",
            "{\"path\": \"nimbus-os.html\", \"edits\": [{\"old_string\": \"a\", \"new_string\": \"b\"}]}\n",
            "Caf\xc3\xa9 men\xc3\xbc \xe2\x9c\x93 \xf0\x9f\x96\xa5 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e text. ",
            "<parameter=path>\nnimbus-os.html\n</parameter>\n<parameter=content>\n<!doctype html>\n</parameter>\n",
            "    if (x !== undefined && y >= 0) { total += x * 2.5; }\n\n",
            "label = cwd.startsWith(home) ? '~/'+cwd.slice(home.length) : cwd;\n",
            "\t\tindent\twith\ttabs and   spaces   \n\n\n",
        };
        std::mt19937 r(7);
        while (synth.size() < 200000) {
            synth += parts[r() % (sizeof(parts) / sizeof(parts[0]))];
        }
    }
    test_fuzz(synth, 3000, 1234);
    if (argc > 2) {
        std::ifstream f(argv[2], std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string corpus = ss.str();
        if (corpus.size() > 10000) {
            test_fuzz(corpus, 3000, 4321);
        } else {
            printf("FAIL: corpus %s too small (%zu bytes)\n", argv[2], corpus.size());
            n_fail++;
        }
    }

    llama_model_free(model);
    llama_backend_free();
    printf("%d checks, %d failed\n", n_check, n_fail);
    return n_fail == 0 ? 0 : 1;
}
