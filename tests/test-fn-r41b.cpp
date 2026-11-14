// [TAG_FN_R41B] CPU unit test of round 41's hot-set state and answer-start rules (src/llama-fn-r41b.h, header only: the
// engine runs the same code):
//   - [TAG_FN_R41_STATEMERGE] the merge: a run of a few decode steps changes little and never drops a stored resident
//     (the night of 13-14 Nov: a state of 182,068 steps and 5,515 residents replaced by test loads of a few dozen steps),
//     a long change of use still takes over, the weights, the run's counts at the decayed counts' scale, the format that
//     v0.33.0's parser reads, the file written through a temporary file and a rename with the previous generation kept
//   - [TAG_FN_R41_STATEKEY] the weights' identity (hard links of one file share it, a copy does not), their names, the key,
//     and the choice of the old state to adopt (the most decode steps of this model)
//   - [TAG_FN_R41_PASSPRIO] on Windows, the page priority rules the switch relies on, on this system: a page read in from
//     the disk by a thread at page priority 2 leaves the working set to the priority-2 standby list (the first the system
//     repurposes), one read by a normal thread to the priority-5 list; a normal thread that faults a low page back in
//     raises it to 5; a low thread never lowers a page that was in RAM
// No GPU, no model. Exit code 0 iff every check passes.

#include "../src/llama-fn-r41b.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <psapi.h>
#endif

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

using hot_layer = fn_r41b::hot_layer;
namespace fs = std::filesystem;

static const char * k_model = "Qwen3.8_Flash_Next";

static std::set<int32_t> res_set(const std::vector<hot_layer> & v) {
    std::set<int32_t> s;
    for (const auto & l : v) {
        for (const int32_t e : l.res) {
            s.insert(l.il * 4096 + e);
        }
    }
    return s;
}

static size_t n_res(const std::vector<hot_layer> & v) {
    size_t n = 0;
    for (const auto & l : v) {
        n += l.res.size();
    }
    return n;
}

// a learned state: n_layer layers of n_exp experts, n_res residents each (the best counts), counts from a skewed draw
static std::vector<hot_layer> make_state(int n_layer, int n_exp, int n_res_l, uint32_t seed, size_t bytes = 3072000) {
    std::mt19937 rng(seed);
    std::vector<hot_layer> out;
    for (int il = 0; il < n_layer; ++il) {
        hot_layer l;
        l.il    = il;
        l.bytes = bytes;
        l.cnt.resize((size_t) n_exp);
        std::vector<int32_t> perm((size_t) n_exp);
        for (int e = 0; e < n_exp; ++e) {
            perm[(size_t) e] = e;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int r = 0; r < n_exp; ++r) {
            l.cnt[(size_t) perm[(size_t) r]] = (float) (40.0 / (1.0 + 0.15*r)); // rank r of the layer's skew
        }
        l.res.assign(perm.begin(), perm.begin() + n_res_l);
        out.push_back(std::move(l));
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------

static void test_meta_format() {
    std::vector<hot_layer> st = make_state(3, 16, 4, 7);
    const std::string text = fn_r41b::format(16, 1234, 0xabcull, k_model, st);
    const std::string hdr  = text.substr(0, text.find('\n'));
    TCHECK(hdr.find("steps=1234") != std::string::npos && hdr.find(" learned=1234") != std::string::npos &&
           hdr.find(" gen=0000000000000abc") != std::string::npos, "header tokens: %s", hdr.c_str());
    // v0.33.0's parser (llama-moe-decay.h, unchanged) reads it
    std::istringstream is(text);
    std::vector<hot_layer> back;
    uint64_t steps = 0;
    std::string why;
    size_t bad = 0;
    TCHECK(llama_moe_hotstate_parse(is, 16, k_model, back, steps, why, &bad), "v0.33.0's parser reads it (%s)", why.c_str());
    TCHECK(steps == 1234 && bad == 0 && back.size() == 3, "steps %llu, %zu layers, %zu bad", (unsigned long long) steps,
            back.size(), bad);
    bool same = back.size() == st.size();
    for (size_t i = 0; same && i < st.size(); ++i) {
        same = back[i].il == st[i].il && back[i].res == st[i].res && back[i].cnt.size() == st[i].cnt.size();
        for (size_t e = 0; same && e < st[i].cnt.size(); ++e) {
            same = std::fabs(back[i].cnt[e] - st[i].cnt[e]) <= 1e-5f*std::max(1.0f, st[i].cnt[e]);
        }
    }
    TCHECK(same, "layers round trip");
    const fn_r41b::state_meta m = fn_r41b::meta_parse(hdr);
    TCHECK(m.ok && m.has_learned && m.learned == 1234 && m.steps == 1234 && m.gen == 0xabcull, "meta of a merged state");
    const fn_r41b::state_meta o = fn_r41b::meta_parse("moehot v1 n_expert=512 layers=48 steps=100670 model=Qwen3.8_Flash_Next");
    TCHECK(o.ok && !o.has_learned && o.learned == 100670 && o.gen == 0, "a v0.33.0 header: learned = its steps");
    TCHECK(!fn_r41b::meta_parse("moeprof v1 n_layer=48").ok, "not a state");
    TCHECK(fn_r41b::new_gen() != 0 && fn_r41b::new_gen() != fn_r41b::new_gen(), "generations differ");
}

static void test_weight() {
    TCHECK(fn_r41b::merge_weight(0, true, 182068, 65536) == 0.0, "no step: weight 0");
    TCHECK(fn_r41b::merge_weight(0, false, 0, 65536) == 0.0, "no step, no state: weight 0 (nothing written)");
    TCHECK(fn_r41b::merge_weight(35, false, 0, 65536) == 1.0, "no stored state: the run is the state");
    TCHECK(fn_r41b::merge_weight(35, true, 0, 65536) == 1.0, "a stored state of no step: the run is the state");
    const double w35 = fn_r41b::merge_weight(35, true, 182068, 65536);
    TCHECK(w35 > 0.0 && w35 < 0.0006, "35 steps against 182,068: %.6f", w35);
    TCHECK(std::fabs(fn_r41b::merge_weight(65536, true, 182068, 65536) - 0.5) < 1e-12, "a horizon's steps: one half");
    TCHECK(std::fabs(fn_r41b::merge_weight(1000, true, 500, 65536) - 1000.0/1500.0) < 1e-12, "a young state counts its steps");
}

// the run's counts land at the scale the decayed counts hold at a pass (the engine's own count and decay calls)
static void test_profile() {
    const int every = 2;
    const float decay = 0.95f;
    std::vector<float>    dcnt(4, 0.0f);
    std::vector<uint32_t> routed(4, 0);
    const uint64_t n = 4000;
    for (uint64_t s = 1; s <= n; ++s) {
        std::vector<int32_t> ids = { 0 };
        if (s % 2 == 0) {
            ids.push_back(1);
        }
        if (s % 10 == 0) {
            ids.push_back(2);
        }
        llama_moe_decay_count(dcnt, ids.data(), (int64_t) ids.size());
        for (const int32_t e : ids) {
            routed[(size_t) e]++;
        }
        if (s % every == 0 && s < n) {
            llama_moe_decay_apply(dcnt, decay);
        }
    }
    std::vector<float> prof;
    fn_r41b::run_profile(routed, n, every, decay, prof);
    TCHECK(std::fabs(prof[0] - 40.0f) < 1e-3f && std::fabs(prof[1] - 20.0f) < 1e-3f && prof[3] == 0.0f,
           "profile %.3f %.3f %.3f %.3f", prof[0], prof[1], prof[2], prof[3]);
    for (int e = 0; e < 3; ++e) {
        TCHECK(std::fabs(prof[(size_t) e] - dcnt[(size_t) e]) <= 0.06f*dcnt[(size_t) e] + 0.5f,
               "expert %d: profile %.2f vs decayed count at a pass %.2f", e, prof[(size_t) e], dcnt[(size_t) e]);
    }
    fn_r41b::run_profile(routed, 0, every, decay, prof);
    TCHECK(prof.size() == 4 && prof[0] == 0.0f, "no step: zero counts");
}

static void test_merge_rules() {
    const std::vector<hot_layer> B = make_state(4, 64, 16, 11);
    const std::vector<hot_layer> R = make_state(4, 64, 12, 12);
    std::vector<hot_layer> out;
    fn_r41b::merge_layers(B, R, 0.0, out);
    bool same = out.size() == B.size();
    for (size_t i = 0; same && i < B.size(); ++i) {
        same = out[i].cnt == B[i].cnt && std::set<int32_t>(out[i].res.begin(), out[i].res.end()) ==
                                          std::set<int32_t>(B[i].res.begin(), B[i].res.end());
    }
    TCHECK(same, "weight 0: the stored state exactly");
    fn_r41b::merge_layers(B, R, 1.0, out);
    same = out.size() == R.size();
    for (size_t i = 0; same && i < R.size(); ++i) {
        same = out[i].cnt == R[i].cnt && std::set<int32_t>(out[i].res.begin(), out[i].res.end()) ==
                                          std::set<int32_t>(R[i].res.begin(), R[i].res.end());
    }
    TCHECK(same, "weight 1: the run exactly");
    // a residents list is ordered by membership, then count
    fn_r41b::merge_layers(B, R, 0.25, out);
    bool order = true;
    for (size_t i = 0; i < out.size(); ++i) {
        std::set<int32_t> bs(B[i].res.begin(), B[i].res.end()), rs(R[i].res.begin(), R[i].res.end());
        auto sc = [&](int32_t e) { return (bs.count(e) ? 0.75 : 0.0) + (rs.count(e) ? 0.25 : 0.0); };
        for (size_t k = 1; k < out[i].res.size(); ++k) {
            const int32_t a = out[i].res[k - 1], b = out[i].res[k];
            order = order && (sc(a) > sc(b) || (sc(a) == sc(b) && out[i].cnt[a] >= out[i].cnt[b]));
        }
        order = order && out[i].res.size() == std::max(B[i].res.size(), R[i].res.size());
        for (const int32_t e : B[i].res) {
            order = order && std::find(out[i].res.begin(), out[i].res.end(), e) != out[i].res.end();
        }
    }
    TCHECK(order, "weight 0.25: every stored resident kept, membership then count, as many as the larger side");
    // a layer on one side only, a layer of another expert size
    std::vector<hot_layer> B2 = B, R2 = R;
    R2.erase(R2.begin() + 1);   // the run lacks layer 1 (not in its hot cache)
    R2[0].bytes = 999;          // the run's layer 0 is another quantization
    hot_layer extra = R[3];
    extra.il = 40;
    R2.push_back(extra);        // a layer the stored state lacks
    fn_r41b::merge_layers(B2, R2, 0.1, out);
    TCHECK(out.size() == 5, "layers: %zu", out.size());
    TCHECK(out.size() == 5 && out[1].il == 1 && out[1].cnt == B[1].cnt && out[1].res == B[1].res, "a layer the run lacks stays as stored");
    TCHECK(out.size() == 5 && out[0].bytes == 999 && out[0].cnt == R[0].cnt, "another expert size: the run's layer");
    TCHECK(out.size() == 5 && out[4].il == 40 && out[4].res == R[3].res, "a new layer: the run's");
}

// the night of 13-14 Nov: a state of 182,068 steps (5,515 residents) and a dozen test loads of a few dozen steps each,
// every one of which saved before the background state load had published its residents
static void test_owner_night() {
    const int n_layer = 48, n_exp = 512, n_res_l = 115;
    const std::vector<hot_layer> start = make_state(n_layer, n_exp, n_res_l, 2026);
    const size_t start_res = n_res(start);
    uint64_t learned = 182068;

    // v0.33.0 (LLAMA_FN_R41_STATEMERGE=0): the run's set is the file
    std::vector<hot_layer> v0330 = start;
    // with the switch: each load reads the merged file of the one before
    std::vector<hot_layer> merged = start;
    std::mt19937 rng(99);
    for (int load = 0; load < 12; ++load) {
        const uint64_t n = 27 + (uint64_t) (rng() % 20);
        std::vector<hot_layer> run = make_state(n_layer, n_exp, 0, 500 + (uint32_t) load);
        for (int il = 0; il < n_layer; ++il) {
            // the residents that had landed when it saved: about half of the loaded ones, plus a few of its own
            const auto & from = v0330[(size_t) il].res;
            for (size_t k = 0; k < from.size() / 2; ++k) {
                run[(size_t) il].res.push_back(from[k]);
            }
            for (int x = 0; x < 3; ++x) {
                run[(size_t) il].res.push_back((int32_t) (rng() % n_exp));
            }
            std::sort(run[(size_t) il].res.begin(), run[(size_t) il].res.end());
            run[(size_t) il].res.erase(std::unique(run[(size_t) il].res.begin(), run[(size_t) il].res.end()), run[(size_t) il].res.end());
        }
        v0330 = run;
        std::vector<hot_layer> runm = run;
        for (int il = 0; il < n_layer; ++il) {
            runm[(size_t) il].res.clear();
            const auto & from = merged[(size_t) il].res;
            for (size_t k = 0; k < from.size() / 2; ++k) {
                runm[(size_t) il].res.push_back(from[k]);
            }
        }
        const double w = fn_r41b::merge_weight(n, true, learned, 65536);
        std::vector<hot_layer> out;
        fn_r41b::merge_layers(merged, runm, w, out);
        merged  = out;
        learned += n;
    }
    const std::set<int32_t> s0 = res_set(start), s1 = res_set(merged);
    size_t kept = 0;
    for (const int32_t k : s0) {
        kept += s1.count(k);
    }
    TCHECK(kept == start_res, "with the switch every stored resident survives 12 short loads: %zu of %zu", kept, start_res);
    // the counts: the sum of the changes against the sum of the counts (every layer), and the residents' own
    double sum_a = 0.0, sum_d = 0.0, max_rel = 0.0;
    for (size_t il = 0; il < start.size(); ++il) {
        for (size_t e = 0; e < start[il].cnt.size(); ++e) {
            const double a = start[il].cnt[e], b = merged[il].cnt[e];
            sum_a += a;
            sum_d += std::fabs(a - b);
        }
        for (const int32_t e : start[il].res) {
            const double a = start[il].cnt[(size_t) e], b = merged[il].cnt[(size_t) e];
            max_rel = std::max(max_rel, std::fabs(a - b) / a);
        }
    }
    TCHECK(sum_d / sum_a < 0.01, "the counts move by < 1 %% in all (%.4f)", sum_d / sum_a);
    TCHECK(max_rel < 0.05, "a resident's count moves by < 5 %% (%.4f)", max_rel);
    TCHECK(learned > 182068 && learned < 182068 + 12*50, "the steps add up: %llu", (unsigned long long) learned);
    // the v0.33.0 rule loses most of them (the failure the switch fixes)
    const std::set<int32_t> sv = res_set(v0330);
    size_t kept_v = 0;
    for (const int32_t k : s0) {
        kept_v += sv.count(k);
    }
    TCHECK(kept_v < start_res / 2, "v0.33.0: %zu of %zu stored residents left after the 12 loads", kept_v, start_res);
    printf("  owner's night: with the merge %zu of %zu residents kept (counts %.2f %% in all, a resident's <= %.2f %%), "
            "v0.33.0 %zu\n", kept, start_res, sum_d / sum_a * 100.0, max_rel*100.0, kept_v);
}

// a lasting change of use takes over: long runs with another set
static void test_adapts() {
    const int n_layer = 8, n_exp = 512, n_res_l = 100;
    std::vector<hot_layer> st = make_state(n_layer, n_exp, n_res_l, 1);
    const std::vector<hot_layer> other = make_state(n_layer, n_exp, n_res_l, 2);
    const std::set<int32_t> so = res_set(other);
    uint64_t learned = 182068;
    std::vector<double> share;
    bool same_size = true;
    for (int k = 0; k < 6; ++k) {
        const uint64_t n = 65536;
        std::vector<hot_layer> out;
        fn_r41b::merge_layers(st, other, fn_r41b::merge_weight(n, true, learned, 65536), out);
        st = out;
        learned += n;
        const std::set<int32_t> ss = res_set(st);
        size_t in = 0;
        for (const int32_t x : ss) {
            in += so.count(x);
        }
        same_size = same_size && ss.size() == so.size();
        share.push_back((double) in / (double) ss.size());
    }
    TCHECK(same_size, "as many residents as before");
    bool rising = true;
    for (size_t k = 1; k < share.size(); ++k) {
        rising = rising && share[k] >= share[k - 1];
    }
    TCHECK(rising && share[0] >= 0.4 && share[2] >= 0.8 && share[5] >= 0.97, "the new use takes over: %.2f %.2f %.2f %.2f %.2f %.2f "
           "after 1-6 runs of the horizon", share[0], share[1], share[2], share[3], share[4], share[5]);
    printf("  a lasting change of use: the new set holds %.0f / %.0f / %.0f %% of the residents after 1 / 3 / 6 runs of 65536 steps\n",
            share[0]*100.0, share[2]*100.0, share[5]*100.0);
}

// ---------------------------------------------------------------------------------------------------------------------

static fs::path make_dir(const char * name) {
    std::error_code ec;
    fs::path d = fs::temp_directory_path(ec) / (std::string("test-fn-r41b-") + name + "-" + std::to_string(fn_r41b::new_gen() & 0xffffff));
    fs::create_directories(d, ec);
    return d;
}

static void test_files() {
    const fs::path d = make_dir("files");
    const std::string p = (d / "state.hotstate").u8string();
    std::string err;
    TCHECK(fn_r41b::write_atomic(p, "one\n", err), "first write (%s)", err.c_str());
    std::string t;
    TCHECK(fn_r41b::read_all(p, t) && t == "one\n", "content");
    TCHECK(!fn_r41b::keep_prev((d / "none.hotstate").u8string(), err), "no file: no .prev");
    TCHECK(fn_r41b::keep_prev(p, err), ".prev written (%s)", err.c_str());
    TCHECK(fn_r41b::write_atomic(p, "two\n", err), "a write over the old file (%s)", err.c_str());
    TCHECK(fn_r41b::read_all(p, t) && t == "two\n", "replaced");
    TCHECK(fn_r41b::read_all(p + ".prev", t) && t == "one\n", "the previous generation kept");
    size_t n_tmp = 0;
    for (const auto & e : fs::directory_iterator(d)) {
        n_tmp += e.path().extension() == ".tmp" ? 1 : 0;
    }
    TCHECK(n_tmp == 0, "no temporary file left");
    int64_t size = -2;
    TCHECK(fn_r41b::read_header(p, &size) == "two" && size == 4, "header and size");
    TCHECK(fn_r41b::read_header((d / "none").u8string(), &size).empty() && size == -1, "no file: size -1");

    // a stored v0.33.0 state, a merged save over it (the engine's sequence: .prev at the first save, then the new file)
    const std::vector<hot_layer> stored = make_state(4, 32, 6, 5);
    TCHECK(fn_r41b::write_atomic(p, llama_moe_hotstate_format(32, 100670, k_model, stored), err), "v0.33.0 state");
    const fn_r41b::state_file s = fn_r41b::read_state(p, 32, k_model);
    TCHECK(s.ok && s.meta.learned == 100670 && s.layers.size() == 4 && s.size > 0, "read: ok %d learned %llu", s.ok ? 1 : 0,
            (unsigned long long) s.meta.learned);
    const fn_r41b::state_file other = fn_r41b::read_state(p, 32, "Qwen3.8 27B");
    TCHECK(!other.ok && other.size > 0, "another model: not usable, but there is a file");
    std::vector<hot_layer> run = make_state(4, 32, 3, 6), out;
    const double w = fn_r41b::merge_weight(40, s.ok, s.meta.learned, 65536);
    fn_r41b::merge_layers(s.layers, run, w, out);
    std::string before;
    fn_r41b::read_all(p, before);
    TCHECK(fn_r41b::keep_prev(p, err) && fn_r41b::write_atomic(p, fn_r41b::format(32, s.meta.learned + 40, fn_r41b::new_gen(),
            k_model, out), err), "merged save");
    TCHECK(fn_r41b::read_all(p + ".prev", t) && t == before, ".prev = the stored state before the save");
    const fn_r41b::state_file s2 = fn_r41b::read_state(p, 32, k_model);
    TCHECK(s2.ok && s2.meta.learned == 100710 && s2.meta.has_learned && n_res(s2.layers) >= n_res(stored), "the merged file: "
            "learned %llu, %zu residents", (unsigned long long) s2.meta.learned, n_res(s2.layers));
    TCHECK(fn_r41b::write_atomic(p, "moehot v1 n_expert=32 layers=1 steps=5 model=Qwen3.8_Flash_Next\nL 0 x\n", err), "corrupt");
    const fn_r41b::state_file s3 = fn_r41b::read_state(p, 32, k_model);
    TCHECK(s3.ok && s3.layers.empty() && s3.n_bad == 1, "a broken layer line is counted (%zu)", s3.n_bad);
    std::error_code ec;
    fs::remove_all(d, ec);
}

static void test_identity_adopt() {
    const fs::path d = make_dir("ident");
    const fs::path a = d / "Model-v3.1-MTPq3-00001-of-00005.gguf";
    const fs::path b = d / "Model-v3.1-MTPq3-512K-2conn-00001-of-00005.gguf";
    const fs::path c = d / "copy.gguf";
    std::string err;
    TCHECK(fn_r41b::write_atomic(a.u8string(), std::string(8192, 'w'), err), "weights file");
    std::error_code ec;
    fs::create_hard_link(a, b, ec);
    TCHECK(!ec, "hard link (%s)", ec.message().c_str());
    fs::copy_file(a, c, ec);
    const fn_r41b::file_id ia = fn_r41b::file_identity(a.u8string()), ib = fn_r41b::file_identity(b.u8string()),
                           ic = fn_r41b::file_identity(c.u8string()), in = fn_r41b::file_identity((d / "none").u8string());
    TCHECK(ia.ok && ib.ok && ic.ok && !in.ok, "identities read");
    TCHECK(ia.vol == ib.vol && ia.idx == ib.idx && ia.nlinks == 2 && ib.nlinks == 2, "hard links: one identity, 2 links");
    TCHECK(!(ia.vol == ic.vol && ia.idx == ic.idx) && ic.nlinks == 1, "a copy: another identity");
    TCHECK(fn_r41b::key_name("Qwen3.8_Flash_Next", ia) == fn_r41b::key_name("Qwen3.8_Flash_Next", ib), "one key for both names");
    TCHECK(fn_r41b::key_name("Qwen3.8_Flash_Next", ia) != fn_r41b::key_name("Qwen3.8_Flash_Next", ic), "another key for the copy");
    const std::string kn = fn_r41b::key_name("Qwen3.8 Flash/Next (x:y)", ia);
    TCHECK(kn.rfind("Qwen3.8_Flash_Next__x_y_-", 0) == 0 && kn.size() > 30 && kn.find(".hotstate") == kn.size() - 9,
           "a key name a folder can hold: %s", kn.c_str());
    const std::vector<std::string> names = fn_r41b::link_names(b.u8string());
#ifdef _WIN32
    TCHECK(names.size() == 2 && names[0] == b.filename().u8string() && names[1] == a.filename().u8string(),
           "the names of the links (%zu)", names.size());
#else
    TCHECK(names.size() == 1 && names[0] == b.filename().u8string(), "the given name");
#endif

    // the old states under the file's names (and two that must not win: another model, a broken file)
    const fs::path hs = d / "hotstate";
    fs::create_directories(hs, ec);
    auto put = [&](const std::string & name, uint64_t steps, const char * model, int n_res_l) {
        (void) fn_r41b::write_atomic((hs / (name + ".hotstate")).u8string(),
                llama_moe_hotstate_format(32, steps, model, make_state(2, 32, n_res_l, (uint32_t) steps)), err);
    };
    put("v3.1-00001-of-00005.gguf", 0, k_model, 0);
    put("512K-1conn-00001-of-00005.gguf", 43351, k_model, 6);
    put("512K-2conn-00001-of-00005.gguf", 11264, k_model, 6);
    put("other-model.gguf", 900000, "Qwen3.8_27B", 6);
    (void) fn_r41b::write_atomic((hs / "broken.gguf.hotstate").u8string(), "not a state\n", err);
    const std::vector<std::string> cand_names = { "v3.1-00001-of-00005.gguf", "512K-2conn-00001-of-00005.gguf",
            "512K-1conn-00001-of-00005.gguf", "other-model.gguf", "broken.gguf", "never-loaded.gguf" };
    const auto cands = fn_r41b::adopt_scan(hs.u8string(), cand_names, 32, k_model);
    TCHECK(cands.size() == 5, "five files found, the missing one left out (%zu)", cands.size());
    const int best = fn_r41b::adopt_pick(cands);
    TCHECK(best >= 0 && fs::u8path(cands[(size_t) best].path).filename().u8string() == "512K-1conn-00001-of-00005.gguf.hotstate" &&
           cands[(size_t) best].learned == 43351 && cands[(size_t) best].n_res == 12, "the most decode steps of this model wins");
    std::vector<fn_r41b::adopt_cand> none(2);
    TCHECK(fn_r41b::adopt_pick(none) == -1, "nothing usable: none");

    // the weights downloaded again (or restored, copied, rebuilt): a new identity of the same name. The key of the earlier
    // identity holds what was learned since the key came in (200,000 steps); the name files are what v0.33.0 left
    fn_r41b::file_id old_id = ia;
    old_id.idx ^= 0x5a5a;
    const std::string old_key = fn_r41b::key_name(k_model, old_id);
    const std::string new_key = (hs / fn_r41b::key_name(k_model, ic)).u8string();
    (void) fn_r41b::write_atomic((hs / old_key).u8string(), fn_r41b::format(32, 200000, 7, k_model, make_state(2, 32, 6, 9)), err);
    (void) fn_r41b::write_atomic((hs / old_key).u8string() + ".prev",
            fn_r41b::format(32, 900000, 8, k_model, make_state(2, 32, 6, 10)), err);         // a .prev is not a key
    (void) fn_r41b::write_atomic((hs / old_key).u8string() + ".1234.tmp", "moehot v1\n", err); // nor a temporary file
    fn_r41b::file_id q_id = ia;
    q_id.idx ^= 0x7777;
    (void) fn_r41b::write_atomic((hs / fn_r41b::key_name(k_model, q_id)).u8string(),
            fn_r41b::format(32, 500000, 9, k_model, make_state(2, 32, 6, 11, 2048000)), err); // another quantization
    fn_r41b::file_id o_id = ia;
    o_id.idx ^= 0x1111;
    (void) fn_r41b::write_atomic((hs / fn_r41b::key_name("Qwen3.8_27B", o_id)).u8string(),
            fn_r41b::format(32, 800000, 10, "Qwen3.8_27B", make_state(2, 32, 6, 12)), err); // another model
    const std::vector<size_t> lb = { 3072000, 3072000 };
    const auto before = fn_r41b::adopt_scan(hs.u8string(), cand_names, 32, k_model, &lb);
    const int  b0     = fn_r41b::adopt_pick(before);
    TCHECK(b0 >= 0 && before[(size_t) b0].learned == 43351, "the file's names alone: the stale name file (%llu steps)",
           b0 >= 0 ? (unsigned long long) before[(size_t) b0].learned : 0ull);
    const std::vector<std::string> peers = fn_r41b::key_peers(hs.u8string(), new_key);
    TCHECK(peers.size() == 2 && peers[0] + ".hotstate" != fn_r41b::key_name(k_model, ic), "two earlier keys of this name (%zu)",
           peers.size());
    std::vector<std::string> all = cand_names;
    all.insert(all.end(), peers.begin(), peers.end());
    const auto after = fn_r41b::adopt_scan(hs.u8string(), all, 32, k_model, &lb);
    const int  b1    = fn_r41b::adopt_pick(after);
    TCHECK(b1 >= 0 && fs::u8path(after[(size_t) b1].path).filename().u8string() == old_key &&
           after[(size_t) b1].learned == 200000, "with the earlier keys: the state learned since the key (%s)",
           b1 >= 0 ? fs::u8path(after[(size_t) b1].path).filename().u8string().c_str() : "none");
    size_t n_other_size = 0;
    for (const auto & c : after) {
        n_other_size += !c.ok && c.why == "another expert size" ? 1 : 0;
    }
    TCHECK(n_other_size == 1, "the key of another quantization is not usable (%zu)", n_other_size);
    TCHECK(fn_r41b::key_peers(hs.u8string(), (hs / "512K-1conn-00001-of-00005.gguf.hotstate").u8string()).empty(),
           "a name file has no key peers");
    TCHECK(fn_r41b::key_peers((d / "none").u8string(), new_key).empty(), "no folder: none");
    fs::remove_all(d, ec);
}

// ---------------------------------------------------------------------------------------------------------------------
#ifdef _WIN32
struct mem_lists {
    ULONG_PTR zero, free_, mod, modnw, bad, by_prio[8], rep[8], modpf;
};

static bool mem_lists_read(mem_lists & l) {
    using nt_qsi_t = LONG (WINAPI *)(ULONG, PVOID, ULONG, PULONG);
    static const nt_qsi_t q = (nt_qsi_t) (void *) GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation");
    ULONG n = 0;
    return q && q(80, &l, sizeof(l), &n) >= 0; // SystemMemoryListInformation
}

// the standby lists by priority that grow when the range leaves the working set (VirtualUnlock on pages that are not
// locked takes them out of it); [p] = pages
static std::vector<long long> trim_delta(const uint8_t * base, size_t bytes) {
    mem_lists a = {}, b = {};
    std::vector<long long> d(8, 0);
    if (!mem_lists_read(a)) {
        return {};
    }
    VirtualUnlock((LPVOID) base, bytes);
    if (!mem_lists_read(b)) {
        return {};
    }
    for (int i = 0; i < 8; ++i) {
        d[(size_t) i] = (long long) b.by_prio[i] - (long long) a.by_prio[i];
    }
    return d;
}

static void test_passprio() {
    TCHECK(fn_r41b::passprio_value(0) == 0 && fn_r41b::passprio_value(-3) == 0 && fn_r41b::passprio_value(2) == 2 &&
           fn_r41b::passprio_value(9) == 4, "switch values");
    // page_prio_follow: lowers, and 0 gives the thread its own value back
    std::thread([]() {
        const int orig = fn_r41b::thread_page_prio();
        fn_r41b::page_prio_follow pp;
        pp.apply(0);
        TCHECK(fn_r41b::thread_page_prio() == orig, "0 changes nothing");
        pp.apply(2);
        TCHECK(fn_r41b::thread_page_prio() == 2, "lowered to 2 (%d)", fn_r41b::thread_page_prio());
        pp.apply(0);
        TCHECK(fn_r41b::thread_page_prio() == orig, "back to %d (%d)", orig, fn_r41b::thread_page_prio());
    }).join();

    // a file whose pages are not in RAM (written without the cache)
    const fs::path d = make_dir("prio");
    const std::wstring path = (d / "pages.bin").wstring();
    const size_t MB = 1 << 20, R = 16*MB, N = 6*R;
    {
        HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH,
                nullptr);
        void * buf = VirtualAlloc(nullptr, MB, MEM_COMMIT, PAGE_READWRITE);
        memset(buf, 0x5a, MB);
        DWORD w = 0;
        bool ok = h != INVALID_HANDLE_VALUE && buf != nullptr;
        for (size_t i = 0; ok && i < N/MB; ++i) {
            ok = WriteFile(h, buf, (DWORD) MB, &w, nullptr) && w == MB;
        }
        TCHECK(ok, "test file written");
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
        }
        VirtualFree(buf, 0, MEM_RELEASE);
    }
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE mh = fh != INVALID_HANDLE_VALUE ? CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
    const uint8_t * p = mh ? (const uint8_t *) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0) : nullptr;
    mem_lists probe = {};
    if (!p || !mem_lists_read(probe)) {
        printf("  page priority rules: skipped (%s)\n", p ? "the memory lists cannot be read here" : "no mapping");
    } else {
        volatile uint64_t sink = 0;
        auto touch = [&](const uint8_t * b) { for (size_t o = 0; o < R; o += 4096) { sink += b[o]; } };
        auto low   = [&](auto fn) { std::thread([&]() { fn_r41b::set_thread_page_prio(2); fn(); }).join(); };
        auto pf    = [&](const uint8_t * b) {
            WIN32_MEMORY_RANGE_ENTRY r = { (PVOID) b, R };
            PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
        };
        // the region's pages land on the list of that priority (other processes move the normal list too: three quarters of
        // the region there and almost none on the low one is the normal case)
        const long long np = (long long) (R / 4096);
        auto at = [&](const std::vector<long long> & dl, int prio) {
            if (dl.empty()) {
                return false;
            }
            return prio == 2 ? dl[2] >= np*9/10 : dl[(size_t) prio] >= np*3/4 && dl[2] <= np/10;
        };
        const uint8_t * A = p, * B = p + R, * C = p + 2*R, * D = p + 3*R, * E = p + 4*R;
        low([&]() { touch(A); });
        TCHECK(at(trim_delta(A, R), 2), "a page read in by a priority-2 thread leaves to the priority-2 standby list");
        touch(B);
        TCHECK(at(trim_delta(B, R), 5), "one read in by a normal thread to the priority-5 list");
        low([&]() { pf(C); touch(C); });
        TCHECK(at(trim_delta(C, R), 2), "prefetched and copied by priority-2 threads (the stream's case): 2");
        low([&]() { pf(D); });
        touch(D);
        TCHECK(at(trim_delta(D, R), 5), "prefetched by a low thread, then used by a normal one: 5");
        touch(A);
        TCHECK(at(trim_delta(A, R), 5), "a low page a normal thread faults back in: raised to 5");
        low([&]() { touch(B); });
        TCHECK(at(trim_delta(B, R), 5), "a normal page a low thread faults back in: stays 5");
        touch(E);
        low([&]() { touch(E); });
        TCHECK(at(trim_delta(E, R), 5), "a normal page in the working set a low thread reads: stays 5");
        printf("  page priority rules: checked on this system (SystemMemoryListInformation)\n");
        (void) sink;
    }
    if (p) {
        UnmapViewOfFile(p);
    }
    if (mh) {
        CloseHandle(mh);
    }
    if (fh != INVALID_HANDLE_VALUE) {
        CloseHandle(fh);
    }
    std::error_code ec;
    fs::remove_all(d, ec);
}
#endif

int main() {
    test_meta_format();
    test_weight();
    test_profile();
    test_merge_rules();
    test_owner_night();
    test_adapts();
    test_files();
    test_identity_adopt();
#ifdef _WIN32
    test_passprio();
#endif
    printf("test-fn-r41b: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
