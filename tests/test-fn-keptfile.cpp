// [TAG_R41_PP_SYSKEEP] [TAG_R41_PP_CONVKEEP] the kept files of the conversation store (tools/server/server-keptfile.h, header
// only): the system blocks of LLAMA_FN_R41_SYSKEEP=2 and the conversations of LLAMA_FN_R41_CONVKEEP=1 that stay on disk
// after the server ends. CPU only, no model: real files in a temporary folder.
//
// Checks:
//  - the store is bounded: beyond the count or the size the oldest file goes, at a scan and before a new file is written,
//    and a file in use is never chosen;
//  - a damaged file (a changed header byte, a file cut short, a changed stored byte), an unfinished one (no header) and a
//    foreign one (not a kept file, another kind, another version) is not read back, is deleted, and a file of the same
//    name can be written again;
//  - a file is tied to the exact weights and chat template: the fingerprint changes with the size or the write time of any
//    weights file, with their number, and with one character of the template; it is the same for the same inputs;
//  - the sum of the stored bytes does not depend on how the bytes are cut into pieces.

#include "../tools/server/server-keptfile.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int g_fail   = 0;
static int g_checks = 0;

#define TCHECK(cond, ...)                                        \
    do {                                                         \
        ++g_checks;                                              \
        if (!(cond)) {                                           \
            ++g_fail;                                            \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, "\n");                               \
        }                                                        \
    } while (0)

namespace fs = std::filesystem;

static std::vector<uint8_t> rnd_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto & b : v) {
        b = (uint8_t) rng();
    }
    return v;
}

// a kept file the way the server lays it out: the tokens, the carry, the stored bytes, then the header (last)
struct made {
    fs::path    path;
    kept_header h;
};

static made make_kept(const fs::path & dir, const std::string & name, uint32_t kind, uint64_t fp, int n_tokens, size_t n_body,
                      uint32_t seed, int age_s, bool with_header = true) {
    made m;
    m.path = dir / name;
    std::memset(&m.h, 0, sizeof(m.h));
    const std::vector<uint8_t> toks  = rnd_bytes((size_t) n_tokens * sizeof(int32_t), seed);
    const std::vector<uint8_t> carry = rnd_bytes(100, seed + 1);
    const std::vector<uint8_t> body  = rnd_bytes(n_body, seed + 2);
    kept_header & h = m.h;
    h.kind        = kind;
    h.fp          = fp;
    h.sys_hash    = kept_fnv(KEPT_FNV_SEED, toks.data(), toks.size());
    h.n_tokens    = n_tokens;
    h.n_ckpt      = 1;
    h.ck[0].n_tokens = n_tokens / 2;
    h.ck[0].size_tgt = n_body / 4;
    h.size_tgt    = n_body - n_body / 4 - n_body / 8;
    h.size_dft    = n_body / 8;
    h.n_bytes     = n_body;
    h.tok_off     = KEPT_SECTOR;
    h.carry_off   = h.tok_off + kept_round_up(toks.size());
    h.carry_bytes = carry.size();
    h.base        = h.carry_off + kept_round_up(carry.size());
    kept_sum s;
    s.add(body.data(), body.size());
    h.body_sum = s.done();
    kept_header_seal(h);

    std::vector<uint8_t> file((size_t) h.base + kept_round_up(n_body), 0);
    if (with_header) {
        std::memcpy(file.data(), &h, sizeof(h));
    }
    std::memcpy(file.data() + h.tok_off, toks.data(), toks.size());
    std::memcpy(file.data() + h.carry_off, carry.data(), carry.size());
    std::memcpy(file.data() + h.base, body.data(), body.size());
    {
        std::ofstream out(m.path, std::ios::binary | std::ios::trunc);
        out.write((const char *) file.data(), (std::streamsize) file.size());
    }
    std::error_code ec;
    fs::last_write_time(m.path, fs::file_time_type::clock::now() - std::chrono::seconds(age_s), ec);
    return m;
}

static void flip_byte(const fs::path & p, uint64_t off) {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    f.seekg((std::streamoff) off);
    char c = 0;
    f.read(&c, 1);
    c = (char) (c ^ 0x40);
    f.seekp((std::streamoff) off);
    f.write(&c, 1);
}

static fs::path new_dir(const char * what) {
    const auto t = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path d = fs::temp_directory_path() / ("test-fn-keptfile-" + std::string(what) + "-" + std::to_string((long long) t));
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d, ec);
    return d;
}

static bool there(const fs::path & p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

// the sum: the same whatever the pieces, another one for another byte or another length
static void test_sum() {
    const std::vector<uint8_t> data = rnd_bytes(200001, 7);
    kept_sum whole;
    whole.add(data.data(), data.size());
    const uint64_t ref = whole.done();
    std::mt19937 rng(11);
    for (int round = 0; round < 50; ++round) {
        kept_sum s;
        size_t   at = 0;
        while (at < data.size()) {
            const size_t k = std::min<size_t>(data.size() - at, round == 0 ? 1 : (size_t) (rng() % 4097));
            s.add(data.data() + at, k);
            at += k;
        }
        TCHECK(s.done() == ref, "round %d: the sum depends on the pieces", round);
    }
    for (size_t off : { (size_t) 0, (size_t) 31, (size_t) 32, (size_t) 100000, data.size() - 1 }) {
        std::vector<uint8_t> d2 = data;
        d2[off] ^= 1;
        kept_sum s;
        s.add(d2.data(), d2.size());
        TCHECK(s.done() != ref, "a changed byte at %zu keeps the sum", off);
    }
    {
        kept_sum s;
        s.add(data.data(), data.size() - 1);
        TCHECK(s.done() != ref, "a shorter stream keeps the sum");
        std::vector<uint8_t> z(64, 0);
        kept_sum a, b;
        a.add(z.data(), 32);
        b.add(z.data(), 64);
        TCHECK(a.done() != b.done(), "zeros of two lengths have one sum");
    }
}

// the header: a good one passes, anything else does not
static void test_header() {
    const fs::path d = new_dir("header");
    const made m = make_kept(d, "sysblock-a.kcs", 1, 0x1111, 3000, 50000, 1, 10);
    const uint64_t size = (uint64_t) fs::file_size(m.path);
    TCHECK(kept_header_ok(m.h, 1, size), "a good header is refused");
    TCHECK(!kept_header_ok(m.h, 2, size), "a system block passes as a conversation");
    TCHECK(!kept_header_ok(m.h, 1, size - KEPT_SECTOR), "a file cut short passes");
    for (size_t off = 0; off < sizeof(kept_header); off += 13) {
        kept_header h = m.h;
        ((uint8_t *) &h)[off] ^= 0x10;
        TCHECK(!kept_header_ok(h, 1, size), "a header with a changed byte at %zu passes", off);
    }
    {
        kept_header h = m.h;
        h.version = 1; // the layout before the sum of the stored bytes
        h.check   = kept_header_check(h);
        TCHECK(!kept_header_ok(h, 1, size), "another version passes");
        h = m.h;
        h.n_bytes += 8; // the parts do not add up
        h.check    = kept_header_check(h);
        TCHECK(!kept_header_ok(h, 1, size), "sizes that do not add up pass");
        h = m.h;
        h.n_tokens = 0;
        h.check    = kept_header_check(h);
        TCHECK(!kept_header_ok(h, 1, size), "no tokens pass");
        h = m.h;
        h.base += KEPT_SECTOR;
        h.check = kept_header_check(h);
        TCHECK(!kept_header_ok(h, 1, size), "a moved base passes");
    }
    std::error_code ec;
    fs::remove_all(d, ec);
}

// the bound: the oldest files go beyond the count and beyond the size
static void test_bound() {
    {
        const fs::path d = new_dir("count");
        std::vector<made> f;
        for (int i = 0; i < 6; ++i) { // f[0] the oldest
            f.push_back(make_kept(d, "sysblock-" + std::to_string(i) + ".kcs", 1, 0x1000 + (i % 2), 2000, 20000, 20 + i, 600 - 100 * i));
        }
        const kept_scan_result r = kept_scan_dir(d.string(), "sysblock-", 1, 4, UINT64_MAX);
        TCHECK(r.n_old == 2 && r.n_bad == 0 && r.files.size() == 4, "count bound: %d old, %d bad, %zu left", r.n_old, r.n_bad, r.files.size());
        TCHECK(!there(f[0].path) && !there(f[1].path), "the two oldest files are still there");
        for (int i = 2; i < 6; ++i) {
            TCHECK(there(f[i].path), "file %d is gone", i);
        }
        TCHECK(r.files.size() == 4 && r.files.front().path.filename() == f[2].path.filename() &&
               r.files.back().path.filename() == f[5].path.filename(), "the files left are not in age order");
        // a second scan changes nothing
        const kept_scan_result r2 = kept_scan_dir(d.string(), "sysblock-", 1, 4, UINT64_MAX);
        TCHECK(r2.n_old == 0 && r2.n_bad == 0 && r2.files.size() == 4, "a second scan deletes again");
        std::error_code ec;
        fs::remove_all(d, ec);
    }
    {
        const fs::path d = new_dir("bytes");
        std::vector<made> f;
        uint64_t total = 0;
        for (int i = 0; i < 5; ++i) {
            f.push_back(make_kept(d, "sysblock-" + std::to_string(i) + ".kcs", 1, 0x2000, 1000, 100000, 40 + i, 500 - 100 * i));
            total += (uint64_t) fs::file_size(f.back().path);
        }
        const uint64_t one = total / 5;
        // room for three and a half files: the two oldest go
        const kept_scan_result r = kept_scan_dir(d.string(), "sysblock-", 1, 100, one * 3 + one / 2);
        TCHECK(r.n_old == 2 && r.files.size() == 3, "size bound: %d old, %zu left", r.n_old, r.files.size());
        TCHECK(!there(f[0].path) && !there(f[1].path) && there(f[2].path) && there(f[4].path), "the size bound did not drop the oldest");
        // before a new file of the same size is written with room for three files: one more of the oldest goes
        std::vector<kept_file> left = kept_list(d.string(), "sysblock-");
        std::vector<size_t> plan = kept_plan_room(left, 100, one * 3 + one / 2, 1, one);
        TCHECK(plan.size() == 1 && plan[0] == 0 && left[0].path.filename() == f[2].path.filename(), "the plan for a new file is not the oldest one");
        // the count the same way: three files, at most three with the new one
        plan = kept_plan_room(left, 3, UINT64_MAX, 1, one);
        TCHECK(plan.size() == 1 && plan[0] == 0, "the count plan for a new file is not the oldest one");
        // enough room: nothing goes
        plan = kept_plan_room(left, 4, one * 5, 1, one);
        TCHECK(plan.empty(), "a file goes although there is room");
        // a file in use is never chosen: the next oldest goes in its place
        left[0].busy = true;
        plan = kept_plan_room(left, 3, UINT64_MAX, 1, one);
        TCHECK(plan.size() == 1 && plan[0] == 1, "a file in use is chosen");
        // a new file bigger than the whole bound: every file that is not in use is planned, and the caller does not keep it
        plan = kept_plan_room(left, 100, one * 2, 1, one * 3);
        TCHECK(plan.size() == 2, "a file too big for the bound: %zu planned", plan.size());
        std::error_code ec;
        fs::remove_all(d, ec);
    }
    {
        // the conversations have their own prefix and kind: a scan of one does not touch the other
        const fs::path d = new_dir("kinds");
        const made s = make_kept(d, "sysblock-s.kcs", 1, 0x3000, 1000, 10000, 60, 50);
        const made c = make_kept(d, "conv-c.kcs", 2, 0x3000, 1000, 10000, 61, 40);
        const kept_scan_result rs = kept_scan_dir(d.string(), "sysblock-", 1, 1, UINT64_MAX);
        const kept_scan_result rc = kept_scan_dir(d.string(), "conv-", 2, 1, UINT64_MAX);
        TCHECK(rs.files.size() == 1 && rc.files.size() == 1 && there(s.path) && there(c.path), "one kind's scan touched the other");
        std::error_code ec;
        fs::remove_all(d, ec);
    }
}

// a damaged, unfinished or foreign file is not read back, is deleted, and is replaced
static void test_damaged() {
    const fs::path d = new_dir("damaged");
    const uint64_t fp = 0xabcdef;
    const made good  = make_kept(d, "sysblock-good.kcs", 1, fp, 3000, 60000, 100, 10);
    const made hdr   = make_kept(d, "sysblock-hdr.kcs", 1, fp, 3000, 60000, 101, 20);
    const made cut   = make_kept(d, "sysblock-cut.kcs", 1, fp, 3000, 60000, 102, 30);
    const made unfin = make_kept(d, "sysblock-unfinished.kcs", 1, fp, 3000, 60000, 103, 40, false);
    const made kind2 = make_kept(d, "sysblock-kind.kcs", 2, fp, 3000, 60000, 104, 50);
    const made body  = make_kept(d, "sysblock-body.kcs", 1, fp, 3000, 60000, 105, 60);
    const made other = make_kept(d, "sysblock-other.kcs", 1, fp + 1, 3000, 60000, 106, 70); // other weights or template
    const fs::path junk  = d / "sysblock-junk.kcs";
    const fs::path tiny  = d / "sysblock-tiny.kcs";
    const fs::path stray = d / "notes.txt"; // not a kept file by its name: never touched
    {
        const std::vector<uint8_t> r = rnd_bytes(30000, 9);
        std::ofstream(junk, std::ios::binary).write((const char *) r.data(), (std::streamsize) r.size());
        std::ofstream(tiny, std::ios::binary).write("R41KEPT1", 8);
        std::ofstream(stray, std::ios::binary).write("x", 1);
    }
    flip_byte(hdr.path, 40);                           // a header byte
    fs::resize_file(cut.path, hdr.h.base + 4096);      // cut in the stored bytes
    flip_byte(body.path, body.h.base + 12345);         // a stored byte: the header is still good

    const kept_scan_result r = kept_scan_dir(d.string(), "sysblock-", 1, 100, UINT64_MAX);
    TCHECK(r.n_bad == 6, "%d bad files found, 6 expected", r.n_bad);
    TCHECK(!there(hdr.path), "a file with a damaged header is still there");
    TCHECK(!there(cut.path), "a file cut short is still there");
    TCHECK(!there(unfin.path), "an unfinished file is still there");
    TCHECK(!there(kind2.path), "a file of another kind is still there");
    TCHECK(!there(junk) && !there(tiny), "a foreign file is still there");
    TCHECK(there(stray), "a file that is not a kept file was deleted");
    TCHECK(there(good.path) && there(body.path) && there(other.path), "a file with a good header was deleted");
    TCHECK(r.files.size() == 3, "%zu files left, 3 expected", r.files.size());

    // what the server does with the files left: its own fingerprint only, and the stored bytes must have the header's sum
    int n_ours = 0, n_whole = 0, n_other = 0;
    for (const auto & f : r.files) {
        if (f.h.fp != fp) {
            n_other++;
            continue;
        }
        n_ours++;
        uint64_t sum = 0;
        const bool whole = kept_body_sum_file(f.path, f.h, sum) && sum == f.h.body_sum;
        if (whole) {
            n_whole++;
            TCHECK(f.path.filename() == good.path.filename(), "the whole file is not the good one");
        } else {
            TCHECK(f.path.filename() == body.path.filename(), "the damaged file is not the one with the changed byte");
            std::error_code ec;
            fs::remove(f.path, ec); // the server deletes it and stores the block again
        }
    }
    TCHECK(n_ours == 2 && n_whole == 1 && n_other == 1, "ours %d, whole %d, other %d", n_ours, n_whole, n_other);
    TCHECK(there(other.path), "a file of another fingerprint was deleted");

    // replaced: files of the same names are written again and are read back
    const made hdr2  = make_kept(d, "sysblock-hdr.kcs", 1, fp, 3000, 60000, 201, 0);
    const made body2 = make_kept(d, "sysblock-body.kcs", 1, fp, 3000, 60000, 205, 0);
    const kept_scan_result r2 = kept_scan_dir(d.string(), "sysblock-", 1, 100, UINT64_MAX);
    TCHECK(r2.n_bad == 0 && r2.files.size() == 4, "after the replacement: %d bad, %zu files", r2.n_bad, r2.files.size());
    int n_ok = 0;
    for (const auto & f : r2.files) {
        uint64_t sum = 0;
        if (f.h.fp == fp && kept_body_sum_file(f.path, f.h, sum) && sum == f.h.body_sum) {
            n_ok++;
        }
    }
    TCHECK(n_ok == 3, "%d files of this fingerprint are whole, 3 expected", n_ok);
    (void) hdr2;
    (void) body2;
    std::error_code ec;
    fs::remove_all(d, ec);
}

// a file is tied to the exact weights and chat template
static void test_fingerprint() {
    kept_identity id;
    id.weights  = { "path D:/m/model-00001-of-00003.gguf", "model-00001-of-00003.gguf 1000/111", "model-00002-of-00003.gguf 2000/222",
                    "model-00003-of-00003.gguf 3000/333", "model X, 35 parameters, 6000 bytes", "draft " };
    id.tmpl     = "{%- for m in messages %}{{ m.content }}{%- endfor %}\n[weights file]\n{{ bos }}\n[jinja]";
    id.settings = "ctx 524288 seq 2 unified 1 k 1 v 1 fa 1 batch 2048 ubatch 512 swa_full 0 spec 3, n_max 3";
    id.build    = "b1 (abc)";
    id.env      = { "LLAMA_A=1", "LLAMA_B=2" };
    const uint64_t ref = kept_fingerprint(id);
    TCHECK(kept_fingerprint(id) == ref, "the fingerprint is not stable");
    {
        kept_identity same = id;
        TCHECK(kept_fingerprint(same) == ref, "a copy has another fingerprint");
    }
    std::vector<uint64_t> seen = { ref };
    auto differs = [&](const kept_identity & x, const char * what) {
        const uint64_t f = kept_fingerprint(x);
        for (const uint64_t s : seen) {
            TCHECK(f != s, "%s: the fingerprint is one seen before", what);
        }
        seen.push_back(f);
    };
    kept_identity x = id;
    x.weights[2] = "model-00002-of-00003.gguf 2001/222";
    differs(x, "the size of the second weights file");
    x = id;
    x.weights[3] = "model-00003-of-00003.gguf 3000/334";
    differs(x, "the write time of the third weights file");
    x = id;
    x.weights.erase(x.weights.begin() + 3);
    differs(x, "a weights file less");
    x = id;
    x.weights[0] = "path D:/m2/model-00001-of-00003.gguf";
    differs(x, "the folder of the weights");
    x = id;
    x.weights[5] = "draft d.gguf 5/5";
    differs(x, "a draft model");
    x = id;
    x.tmpl[10] = x.tmpl[10] == 'm' ? 'n' : 'm';
    differs(x, "one character of the chat template");
    x = id;
    x.tmpl += " ";
    differs(x, "a space after the chat template");
    x = id;
    x.settings = "ctx 262144 seq 2 unified 1 k 1 v 1 fa 1 batch 2048 ubatch 512 swa_full 0 spec 3, n_max 3";
    differs(x, "the context size");
    x = id;
    x.build = "b2 (abd)";
    differs(x, "the build");
    x = id;
    x.env[1] = "LLAMA_B=3";
    differs(x, "an engine variable");
    // the parts do not run into each other
    x = id;
    x.tmpl     = "";
    x.settings = id.tmpl + id.settings;
    differs(x, "the template moved into the settings");
    x = id;
    x.weights.push_back(id.tmpl);
    x.tmpl = "";
    differs(x, "the template moved into the weights");
    x = id;
    x.env.clear();
    x.build = id.build + "LLAMA_A=1" + "LLAMA_B=2";
    differs(x, "the variables moved into the build");
}

int main() {
    test_sum();
    test_header();
    test_bound();
    test_damaged();
    test_fingerprint();
    printf("test-fn-keptfile: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
