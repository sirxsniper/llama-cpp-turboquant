// [TAG_FN_L22] round 22 P2 unit tests (CPU only):
//   - [TAG_FN_L22_TOKCACHE] common/tokcache.h gives the ids of a full llama_tokenize for appended, edited, cut and
//     random texts (real vocabulary: models/ggml-vocab-qwen35.gguf, found from the exe or the working directory, or
//     argv[1]; skipped when absent)
//   - tools/mtmd/mtmd-l22.h: parallel_for, the content hash, the encoder output cache, the lend size estimate
//   - [TAG_V0262_TOKCACHE] the fragment probe (a toy tokenizer with and without the metaspace "first" rule) and the
//     vocabulary registry (one entry per vocabulary, at most VOCAB_INFO_MAX, a held entry stays valid)

#include "llama.h"
#include "common.h"
#include "tokcache.h"
#include "../tools/mtmd/mtmd-l22.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) do { if (cond) { g_pass++; } else { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

//
// tokenizer cache
//

static std::string find_vocab(int argc, char ** argv) {
    namespace fs = std::filesystem;
    if (argc > 1 && fs::exists(argv[1])) {
        return argv[1];
    }
    const char * rel = "models/ggml-vocab-qwen35.gguf";
    std::vector<fs::path> bases = { fs::current_path() };
    if (argc > 0 && argv[0] != nullptr) {
        std::error_code ec;
        fs::path exe = fs::absolute(fs::path(argv[0]), ec);
        if (!ec) {
            bases.push_back(exe.parent_path());
        }
    }
    for (auto b : bases) {
        for (int up = 0; up < 6; ++up) {
            fs::path p = b / rel;
            if (fs::exists(p)) {
                return p.string();
            }
            if (!b.has_parent_path() || b.parent_path() == b) {
                break;
            }
            b = b.parent_path();
        }
    }
    return "";
}

static std::string chat_turn(const char * role, const std::string & content) {
    return "<|im_start|>" + std::string(role) + "\n" + content + "<|im_end|>\n";
}

static std::string body(std::mt19937 & rng, int n_lines) {
    static const char * words[] = {
        "the", "function", "returns", "a", "value", "of", "type", "int", "when", "called", "with", "arguments", "x", "y",
        "{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.cpp\"}}", "0.85", "48.858844", "  indented", "\t",
        "unicode: \xc3\xa9\xc3\xa8 \xe4\xb8\xad\xe6\x96\x87 \xf0\x9f\x98\x80", "<think>", "</think>", "<tool_call>", "</tool_call>",
        "if (a < b) { return a; }", "    for i in range(10):", "\n", "\n\n", "...", "--", "https://example.com/a?b=c",
    };
    std::uniform_int_distribution<int> nw(3, 14);
    std::uniform_int_distribution<int> wi(0, (int) (sizeof(words)/sizeof(words[0])) - 1);
    std::string s;
    for (int l = 0; l < n_lines; ++l) {
        const int n = nw(rng);
        for (int k = 0; k < n; ++k) {
            s += words[wi(rng)];
            s += ' ';
        }
        s += '\n';
    }
    return s;
}

static bool same_as_full(const llama_vocab * vocab, common_tokcache & c, const std::string & text, bool add_special,
        common_tokcache_stats * st_out = nullptr) {
    common_tokcache_stats st;
    const auto got  = common_tokcache_tokenize(c, vocab, text, add_special, true, &st);
    const auto want = common_tokenize(vocab, text, add_special, true);
    if (st_out) {
        *st_out = st;
    }
    return got == want;
}

static void test_tokcache(const std::string & path) {
    if (path.empty()) {
        printf("tokcache: models/ggml-vocab-qwen35.gguf not found, skipped\n");
        return;
    }
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        CHECK(false, "cannot load %s", path.c_str());
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    printf("tokcache: %s, %d tokens, type %d\n", path.c_str(), llama_vocab_n_tokens(vocab), (int) llama_vocab_type(vocab));

    std::mt19937 rng(1234);
    common_tokcache c(4, (size_t) 64 << 20, 64);

    // a growing conversation: every new request re-sends the old text plus a turn
    std::string conv = chat_turn("system", "You are a helpful assistant.\n" + body(rng, 30));
    int n_prefix = 0;
    for (int turn = 0; turn < 12; ++turn) {
        conv += chat_turn(turn % 2 ? "assistant" : "user", body(rng, 10 + turn * 3));
        const std::string req = conv + "<|im_start|>assistant\n";
        common_tokcache_stats st;
        CHECK(same_as_full(vocab, c, req, true, &st), "growing conversation, turn %d", turn);
        n_prefix += st.path == 2 ? 1 : 0;
    }
    CHECK(n_prefix >= 10, "prefix re-use on appended turns: %d of 12", n_prefix);

    // [TAG_V0262_TOKCACHE] this vocabulary passes the fragment probe, and one registry entry serves every call
    {
        const auto a = common_tokcache_impl::get_info(vocab);
        const auto b = common_tokcache_impl::get_info(vocab);
        CHECK(a && a->ok, "the qwen35 vocabulary passes the fragment probe");
        CHECK(a == b, "one registry entry per vocabulary");
    }

    // the same text again: exact re-use
    {
        common_tokcache_stats st;
        const std::string req = conv + "<|im_start|>assistant\n";
        CHECK(same_as_full(vocab, c, req, true, &st), "exact re-send");
        CHECK(st.path == 1, "exact re-send path %d", st.path);
    }

    // an edit deep in the history (Kyro trims old pictures / tool results)
    {
        std::string edited = conv;
        const size_t at = edited.size() / 3;
        edited.insert(at, "[an edit in the history]");
        CHECK(same_as_full(vocab, c, edited, true), "edit at 1/3");
        edited = conv.substr(0, conv.size() / 2) + chat_turn("user", "a cut conversation");
        CHECK(same_as_full(vocab, c, edited, true), "cut at 1/2");
    }

    // the new text ends inside a special token of the old one, or splits one
    {
        const std::string base = chat_turn("user", body(rng, 20)) + chat_turn("assistant", body(rng, 20));
        CHECK(same_as_full(vocab, c, base, false), "base");
        const size_t p = base.rfind("<|im_end|>");
        for (size_t k = 1; k < 10; ++k) {
            const std::string cut = base.substr(0, p + k);
            CHECK(same_as_full(vocab, c, cut, false), "cut inside <|im_end|> at %zu", k);
            const std::string split = base.substr(0, p + k) + " " + base.substr(p + k);
            CHECK(same_as_full(vocab, c, split, false), "split <|im_end|> at %zu", k);
        }
    }

    // random edits of recent texts: insert / delete / replace, also with special-token text and partial special tokens
    {
        static const char * ins[] = { "<|im_end|>", "<|im_start|>", "<|im_", "end|>", "<think>", "</think>", "<tool_call>\n", " ",
                                      "\n", "abc", "\xc3\xa9", "0.5", "<", "|", ">" };
        std::vector<std::string> recent = { conv };
        std::uniform_int_distribution<int> op(0, 3);
        std::uniform_int_distribution<int> pick_ins(0, (int) (sizeof(ins)/sizeof(ins[0])) - 1);
        int bad = 0;
        for (int it = 0; it < 300; ++it) {
            std::string s = recent[std::uniform_int_distribution<size_t>(0, recent.size() - 1)(rng)];
            const size_t at = std::uniform_int_distribution<size_t>(0, s.size())(rng);
            switch (op(rng)) {
                case 0: s.insert(at, ins[pick_ins(rng)]); break;
                case 1: s.erase(at, std::min<size_t>(s.size() - at, std::uniform_int_distribution<size_t>(1, 40)(rng))); break;
                case 2: s = s.substr(0, at); break;
                default: s += chat_turn("user", body(rng, 3)); break;
            }
            const bool add_special = (it % 7) == 0;
            if (!same_as_full(vocab, c, s, add_special)) {
                bad++;
                if (bad < 5) {
                    fprintf(stderr, "random edit %d (op at %zu, %zu bytes) differs from the full encode\n", it, at, s.size());
                }
            }
            recent.push_back(s);
            if (recent.size() > 6) {
                recent.erase(recent.begin());
            }
        }
        CHECK(bad == 0, "random edits: %d of 300 differ", bad);
    }

    // short texts bypass the cache, the verify mode compares and keeps the cache on
    {
        common_tokcache v(4, (size_t) 64 << 20, 64);
        v.n_verify.store(8);
        std::string s = chat_turn("system", body(rng, 40));
        for (int k = 0; k < 6; ++k) {
            s += chat_turn("user", body(rng, 5));
            CHECK(same_as_full(vocab, v, s, false), "verify mode, step %d", k);
        }
        CHECK(!v.off.load(), "verify mode found no mismatch");
        common_tokcache_stats st;
        same_as_full(vocab, v, "short", false, &st);
        CHECK(st.path == 3, "short text bypasses the cache (path %d)", st.path);
    }

    // a long conversation (~100K+ tokens) re-sent with one more turn: equal ids, and the time of both paths
    {
        common_tokcache big(4, (size_t) 64 << 20, 4096);
        std::string s = chat_turn("system", body(rng, 200));
        while (s.size() < 450000) {
            s += chat_turn("user", body(rng, 40));
            s += chat_turn("assistant", "<think>\n" + body(rng, 20) + "</think>\n" + body(rng, 60));
        }
        const int64_t t0 = ggml_time_us();
        const auto full0 = common_tokenize(vocab, s, false, true);
        const int64_t t1 = ggml_time_us();
        CHECK(common_tokcache_tokenize(big, vocab, s, false, true) == full0, "long text, first encode");
        const std::string s2 = s + chat_turn("user", body(rng, 10)) + "<|im_start|>assistant\n";
        const int64_t t2 = ggml_time_us();
        common_tokcache_stats st;
        const auto cached = common_tokcache_tokenize(big, vocab, s2, false, true, &st);
        const int64_t t3 = ggml_time_us();
        const auto full2 = common_tokenize(vocab, s2, false, true);
        CHECK(cached == full2, "long text re-send with one more turn");
        CHECK(st.path == 2, "long text re-send takes the prefix path (%d)", st.path);
        printf("tokcache: %zu bytes, %zu ids: full encode %.1f ms, re-send with the cache %.1f ms (re-used %zu ids)\n",
               s2.size(), full2.size(), (t1 - t0)/1000.0, (t3 - t2)/1000.0, st.reused_ids);
    }

    // threads: lookups and inserts at the same time
    {
        common_tokcache mt(4, (size_t) 64 << 20, 64);
        std::vector<std::string> texts;
        std::string s = chat_turn("system", body(rng, 50));
        for (int k = 0; k < 8; ++k) {
            s += chat_turn("user", body(rng, 4));
            texts.push_back(s);
        }
        std::atomic<int> diffs{0};
        mtmd_l22_parallel_for(32, 4, [&](int i) {
            const std::string & tx = texts[(size_t) i % texts.size()];
            if (common_tokcache_tokenize(mt, vocab, tx, false, true) != common_tokenize(vocab, tx, false, true)) {
                diffs++;
            }
        });
        CHECK(diffs.load() == 0, "threaded lookups: %d differ", diffs.load());
    }

    // [TAG_V0262_TOKCACHE] reloads in one process (--sleep-idle-seconds) give new vocabulary addresses: the registry keeps
    // at most VOCAB_INFO_MAX entries, an entry a caller still holds stays valid, and a dropped vocabulary is built again
    {
        const auto held = common_tokcache_impl::get_info(vocab);
        std::vector<llama_model *> more;
        llama_model_params mp2 = llama_model_default_params();
        mp2.vocab_only = true;
        for (size_t k = 0; k < common_tokcache_impl::VOCAB_INFO_MAX + 1; ++k) {
            llama_model * m = llama_model_load_from_file(path.c_str(), mp2);
            if (m == nullptr) {
                break;
            }
            more.push_back(m);
            CHECK(common_tokcache_impl::get_info(llama_model_get_vocab(m))->ok, "reload %zu: tables built", k);
        }
        CHECK(more.size() == common_tokcache_impl::VOCAB_INFO_MAX + 1, "reloads: %zu", more.size());
        CHECK(common_tokcache_impl::registry_size() <= common_tokcache_impl::VOCAB_INFO_MAX, "registry size %zu",
              common_tokcache_impl::registry_size());
        CHECK(held->vocab == vocab && held->ok && !held->pieces.empty(), "a held entry stays valid after it is dropped");
        common_tokcache again(4, (size_t) 64 << 20, 64);
        const std::string tx = chat_turn("system", body(rng, 20)) + chat_turn("user", body(rng, 4));
        CHECK(same_as_full(vocab, again, tx, false), "the first vocabulary after its entry was dropped");
        for (llama_model * m : more) {
            llama_model_free(m);
        }
    }

    llama_model_free(model);
}

// [TAG_V0262_TOKCACHE] a toy tokenizer: "<S>" is a special token (id 1000), every other byte is its own id. With `first`
// a raw fragment at the start of the input that does not start with a space gets id 999 in front, as upstream's
// metaspace prepend_scheme "first" does with U+2581
static std::vector<llama_token> toy_tokenize(const std::string & t, bool first) {
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < t.size()) {
        if (t.compare(i, 3, "<S>") == 0) {
            out.push_back(1000);
            i += 3;
            continue;
        }
        size_t j = t.find("<S>", i);
        j = j == std::string::npos ? t.size() : j;
        if (first && i == 0 && t[0] != ' ') {
            out.push_back(999);
        }
        for (; i < j; ++i) {
            out.push_back((llama_token) (unsigned char) t[i]);
        }
    }
    return out;
}

static void test_tokcache_probe() {
    const auto plain = [](const std::string & t) { return toy_tokenize(t, false); };
    const auto first = [](const std::string & t) { return toy_tokenize(t, true); };
    CHECK(common_tokcache_impl::fragment_local("<S>", plain), "probe: fragments encode alone as in the whole text");
    CHECK(!common_tokcache_impl::fragment_local("<S>", first), "probe: the metaspace \"first\" rule is found");
    // what the probe guards: text after the special token, encoded alone, differs from the full encode there
    const std::string full = "x<S>Assistant\nhi";
    std::vector<llama_token> cut = first("x<S>");
    const auto rest = first("Assistant\nhi");
    cut.insert(cut.end(), rest.begin(), rest.end());
    CHECK(cut != first(full), "probe: the \"first\" rule changes a cut encode");
}

//
// mtmd-l22.h
//

static void test_parallel_for() {
    std::atomic<long long> sum{0};
    mtmd_l22_parallel_for(1000, 4, [&](int i) { sum += i; });
    CHECK(sum.load() == 499500, "parallel_for sum %lld", sum.load());
    std::vector<int> seen(37, 0);
    mtmd_l22_parallel_for(37, 8, [&](int i) { seen[(size_t) i]++; });
    bool once = true;
    for (int v : seen) {
        once = once && v == 1;
    }
    CHECK(once, "parallel_for runs every index once");
    bool threw = false;
    try {
        mtmd_l22_parallel_for(50, 4, [&](int i) { if (i == 17 || i == 31) { throw std::runtime_error(std::to_string(i)); } });
    } catch (const std::runtime_error & e) {
        threw = std::string(e.what()) == "17";
    }
    CHECK(threw, "parallel_for rethrows the lowest index's exception");
    int n1 = 0;
    mtmd_l22_parallel_for(5, 1, [&](int) { n1++; });
    CHECK(n1 == 5, "parallel_for serial");
}

static void test_hash() {
    std::vector<float> a(1000), b;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (float) i * 0.25f;
    }
    b = a;
    mtmd_l22_hash128 ha, hb;
    ha.add(a.data(), a.size() * sizeof(float));
    hb.add(b.data(), b.size() * sizeof(float));
    CHECK(ha == hb, "same data, same hash");
    b[500] += 1e-3f;
    mtmd_l22_hash128 hc;
    hc.add(b.data(), b.size() * sizeof(float));
    CHECK(!(ha == hc), "one value differs, the hash differs");
    mtmd_l22_hash128 hd;
    hd.add(a.data(), (a.size() - 1) * sizeof(float));
    CHECK(!(ha == hd), "shorter data, another hash");
    mtmd_l22_hash128 he, hf;
    const char x[3] = { 1, 2, 3 };
    he.add(x, 3);
    hf.add(x, 2);
    CHECK(!(he == hf), "tail bytes count");
}

static void test_embd_cache() {
    mtmd_l22_embd_cache c;
    c.n_max = 3;
    c.bytes_max = 1000 * sizeof(float);
    auto key = [](int i) { mtmd_l22_hash128 h; h.add_u64((uint64_t) i); return h; };
    std::vector<float> v(100, 1.0f), out;
    CHECK(!c.get(key(1), 100, out), "empty cache misses");
    c.put(key(1), v.data(), v.size());
    CHECK(c.get(key(1), 100, out) && out == v, "hit returns the data");
    CHECK(!c.get(key(1), 99, out), "another size misses");
    v.assign(100, 2.0f);
    c.put(key(2), v.data(), v.size());
    c.put(key(3), v.data(), v.size());
    c.get(key(1), 100, out); // 1 becomes the most recent
    c.put(key(4), v.data(), v.size()); // drops 2
    CHECK(c.get(key(1), 100, out), "recently used entry kept");
    CHECK(!c.get(key(2), 100, out), "least recently used entry dropped");
    std::vector<float> big(800, 3.0f);
    c.put(key(5), big.data(), big.size()); // byte cap: 800 + 100s > 1000
    CHECK(c.bytes <= c.bytes_max, "byte cap kept (%zu)", c.bytes);
    CHECK(c.get(key(5), 800, out) && out == big, "the new big entry is there");
    std::vector<float> huge(2000, 4.0f);
    c.put(key(6), huge.data(), huge.size());
    CHECK(!c.get(key(6), 2000, out), "an entry above the cap is not cached");
    mtmd_l22_embd_cache off;
    off.put(key(1), v.data(), v.size());
    CHECK(!off.get(key(1), 100, out), "n_max 0 caches nothing");
}

static void test_lend_size() {
    const size_t MiB = 1048576;
    mtmd_l22_lend_size s;
    s.c_w = 486 * MiB;
    CHECK(s.estimate(100000, nullptr) == s.c_w, "off: always the warm-up bytes");
    s.on    = true;
    s.n_w   = 8464;
    s.n_max = 16384;
    s.c_max = 940 * MiB;
    CHECK(s.estimate(1000, nullptr) == s.c_w, "small picture: the warm-up bytes");
    CHECK(s.estimate(8464, nullptr) == s.c_w, "warm-up size: the warm-up bytes");
    const size_t e12 = s.estimate(12000, nullptr), e16 = s.estimate(16384, nullptr);
    CHECK(e12 > s.c_w && e12 < e16, "between: above the warm-up, below the largest");
    CHECK(e16 >= s.c_max, "largest picture: at least its measured bytes");
    CHECK(s.estimate(20000, nullptr) > e16, "beyond the largest: extrapolated");
    s.record(1, 2560, 1600, 700 * MiB);
    const size_t * ex = s.find(1, 2560, 1600);
    CHECK(ex != nullptr && *ex == 700 * MiB, "a recorded size is found");
    CHECK(s.estimate(16000, ex) == 700 * MiB, "an exact size wins");
    s.record(1, 2560, 1600, 600 * MiB);
    CHECK(*s.find(1, 2560, 1600) == 700 * MiB, "record keeps the larger size");
    const size_t small = 100 * MiB;
    CHECK(s.estimate(16000, &small) == s.c_w, "never below the warm-up bytes");
    CHECK(s.find(2, 2560, 1600) == nullptr, "another entry count is another size");
}

int main(int argc, char ** argv) {
    llama_backend_init();

    test_parallel_for();
    test_hash();
    test_embd_cache();
    test_lend_size();
    test_tokcache_probe();
    test_tokcache(find_vocab(argc, argv));

    llama_backend_free();
    printf("test-fn-l22-p2: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
