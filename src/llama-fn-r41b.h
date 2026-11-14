#pragma once

// [TAG_FN_R41B] round 41, the learned hot set and the answer's start: the rules of three switches, header only so the unit
// test (tests/test-fn-r41b.cpp) runs the same code as the engine. Nothing here changes a value the model computes.
//
//   LLAMA_FN_R41_STATEKEY=1        [TAG_FN_R41_STATEKEY] (the profile: 0 = v0.33.0) LLAMA_MOE_HOT_STATE=cache keys
//                                  the learned set by the weights, not by the file name: names that are hard links of one
//                                  file (the same volume serial and file index) share one state, <model name>-<volume>-
//                                  <file index>.hotstate. When the identity cannot be read, the file name as before. At the
//                                  first load under the new key the best state among the file's old names and the keys of
//                                  earlier files of the same model name (a download again, a copy: a new file index) is
//                                  copied to it, the one learned over the most decode steps; nothing is deleted
//   LLAMA_FN_R41_STATEMERGE=65536  [TAG_FN_R41_STATEMERGE] (the profile: 0 = v0.33.0, the run's set replaces the file)
//                                  a save merges what this run learned into the stored state with the weight
//                                  n / (n + min(L, H)): n = the run's decode steps, L = the steps the stored state was
//                                  learned over, H = the value (a horizon in decode steps). The counts are mixed by that
//                                  weight (the run's side: how often each expert was routed over the whole run, at the
//                                  scale of the decayed counts), the residents by their weighted membership. The stored
//                                  state is the one read at the start, so saves within a run never compound. The previous
//                                  generation stays as <state>.prev, the file is replaced through a temporary file and a
//                                  rename, a run without a decode step writes nothing. The format stays "moehot v1"
//                                  (v0.33.0 reads it: learned= and gen= are header tokens its parser skips)
//   LLAMA_FN_R41_PASSPRIO=2        [TAG_FN_R41_PASSPRIO] (the Flash-Next profile keeps 0 = v0.33.0 until a GPU arm has
//                                  measured it) the prompt stream's read-ahead, copy and streamer threads run at this page
//                                  priority (1-4, below normal 5): the pages a prompt pass reads in from the disk are the
//                                  first the system takes back, before the decode's own. A page a normal thread faults in
//                                  again gets 5; a lower thread never lowers a page that was in RAM (the unit test checks
//                                  these rules on the system it runs on)

#include "llama-moe-decay.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <io.h>
#else
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace fn_r41b {

using hot_layer = llama_moe_hotstate_layer;

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_STATEMERGE] the header's tokens: steps= (v0.33.0: the decode steps of the run that wrote it), learned= (the
// steps the state was learned over, every run that merged into it) and gen= (one value per write)
struct state_meta {
    bool     ok          = false; // a moehot v1 header
    uint64_t steps       = 0;
    uint64_t learned     = 0;     // learned=, else steps= (a v0.33.0 file knows only its last run)
    bool     has_learned = false;
    uint64_t gen         = 0;
};

inline state_meta meta_parse(const std::string & header) {
    state_meta m;
    if (header.rfind("moehot v1", 0) != 0) {
        return m;
    }
    m.ok = true;
    std::istringstream hs(header);
    std::string tok;
    while (hs >> tok) {
        if (tok.rfind("steps=", 0) == 0) {
            m.steps = strtoull(tok.c_str() + 6, nullptr, 10);
        } else if (tok.rfind("learned=", 0) == 0) {
            m.learned     = strtoull(tok.c_str() + 8, nullptr, 10);
            m.has_learned = true;
        } else if (tok.rfind("gen=", 0) == 0) {
            m.gen = strtoull(tok.c_str() + 4, nullptr, 16);
        }
    }
    if (!m.has_learned) {
        m.learned = m.steps;
    }
    return m;
}

// the file text of a merged state: v0.33.0's format with steps = learned, then the two tokens at the end of its header line
inline std::string format(int64_t n_expert, uint64_t learned, uint64_t gen, const std::string & model,
        const std::vector<hot_layer> & layers) {
    std::string text = llama_moe_hotstate_format(n_expert, learned, model, layers);
    char buf[64];
    snprintf(buf, sizeof(buf), " learned=%llu gen=%016llx", (unsigned long long) learned, (unsigned long long) gen);
    const size_t nl = text.find('\n');
    text.insert(nl == std::string::npos ? text.size() : nl, buf);
    return text;
}

inline uint64_t new_gen() {
    std::random_device rd;
    const uint64_t a = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
    const uint64_t t = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t g = a ^ (t * 0x9E3779B97F4A7C15ull);
    return g != 0 ? g : 1;
}

// the run's weight in the merge: 0 without a decode step, 1 without a stored state (or one learned over no step),
// else n / (n + min(learned, horizon))
inline double merge_weight(uint64_t n_run, bool base_ok, uint64_t learned, uint64_t horizon) {
    if (n_run == 0) {
        return 0.0;
    }
    if (!base_ok || learned == 0) {
        return 1.0;
    }
    const double h = (double) std::min<uint64_t>(learned, std::max<uint64_t>(horizon, 1));
    return (double) n_run / ((double) n_run + h);
}

// how often each expert was routed over the run (decode steps that routed it) at the scale of the decayed counts: an
// expert routed at the rate r holds r x every / (1 - decay) when the pass reads it (each step adds 1, each pass of `every`
// steps multiplies by decay)
inline void run_profile(const std::vector<uint32_t> & routed, uint64_t n_steps, int every, float decay, std::vector<float> & out) {
    out.assign(routed.size(), 0.0f);
    if (n_steps == 0 || !(decay > 0.0f && decay < 1.0f)) {
        return;
    }
    const double s = (double) std::max(1, every) / (1.0 - (double) decay) / (double) n_steps;
    for (size_t e = 0; e < routed.size(); ++e) {
        out[e] = (float) ((double) routed[e] * s);
    }
}

// the residents of a merged layer: every expert resident on either side, by its weighted membership ((1 - w) for the
// stored side, w for the run's), then by its merged count; as many as the larger side holds (a start with a larger hot
// set than the run's loads the stored side's rest after the run's own). A side of weight 0 adds nothing.
inline std::vector<int32_t> merge_res(const std::vector<int32_t> & base, const std::vector<int32_t> & run,
        const std::vector<float> & cnt, double w) {
    const size_t n_exp = cnt.size();
    std::vector<double>  score(n_exp, 0.0);
    std::vector<uint8_t> in_b(n_exp, 0), in_r(n_exp, 0);
    std::vector<int32_t> cand;
    for (const int32_t e : base) {
        if (e >= 0 && (size_t) e < n_exp && !in_b[e]) {
            in_b[e] = 1;
            score[e] += 1.0 - w;
        }
    }
    for (const int32_t e : run) {
        if (e >= 0 && (size_t) e < n_exp && !in_r[e]) {
            in_r[e] = 1;
            score[e] += w;
        }
    }
    for (size_t e = 0; e < n_exp; ++e) {
        if ((in_b[e] || in_r[e]) && score[e] > 0.0) {
            cand.push_back((int32_t) e);
        }
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) {
        if (score[a] != score[b]) {
            return score[a] > score[b];
        }
        return cnt[a] > cnt[b];
    });
    const size_t k = std::max(base.size(), run.size());
    if (cand.size() > k) {
        cand.resize(k);
    }
    return cand;
}

// the stored layers and the run's merged with the run's weight w: a layer on one side only is taken as it is; a layer of
// another expert size (another quantization) on the stored side gives way to the run's
inline void merge_layers(const std::vector<hot_layer> & base, const std::vector<hot_layer> & run, double w,
        std::vector<hot_layer> & out) {
    out.clear();
    auto find = [](const std::vector<hot_layer> & v, int il) -> const hot_layer * {
        for (const auto & l : v) {
            if (l.il == il) {
                return &l;
            }
        }
        return nullptr;
    };
    for (const auto & b : base) {
        const hot_layer * r = find(run, b.il);
        if (r == nullptr || w <= 0.0) {
            out.push_back(b);
            continue;
        }
        if (r->bytes != b.bytes || r->cnt.size() != b.cnt.size()) {
            out.push_back(*r);
            continue;
        }
        hot_layer m;
        m.il    = b.il;
        m.bytes = b.bytes;
        m.cnt.resize(b.cnt.size());
        for (size_t e = 0; e < b.cnt.size(); ++e) {
            m.cnt[e] = w >= 1.0 ? r->cnt[e] : (float) ((1.0 - w)*(double) b.cnt[e] + w*(double) r->cnt[e]);
        }
        m.res = merge_res(b.res, r->res, m.cnt, w);
        out.push_back(std::move(m));
    }
    for (const auto & r : run) {
        if (find(base, r.il) == nullptr) {
            out.push_back(r);
        }
    }
}

// the residents kept from the stored side and the run's that are new (for the log)
inline void merge_counts(const std::vector<hot_layer> & base, const std::vector<hot_layer> & out, size_t & n_res,
        size_t & n_kept, size_t & n_base) {
    n_res = n_kept = n_base = 0;
    for (const auto & m : out) {
        n_res += m.res.size();
        for (const auto & b : base) {
            if (b.il != m.il) {
                continue;
            }
            n_base += b.res.size();
            std::vector<uint8_t> in(m.cnt.size() + 1, 0);
            for (const int32_t e : m.res) {
                if (e >= 0 && (size_t) e < in.size()) {
                    in[e] = 1;
                }
            }
            for (const int32_t e : b.res) {
                n_kept += e >= 0 && (size_t) e < in.size() && in[e] ? 1 : 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// files

inline std::filesystem::path u8p(const std::string & s) {
    return std::filesystem::u8path(s);
}

inline bool read_all(const std::string & path, std::string & out) {
    std::ifstream f(u8p(path), std::ios::binary);
    if (!f) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// the first line of a file ("" when there is none) and its size (-1: no file)
inline std::string read_header(const std::string & path, int64_t * size = nullptr) {
    if (size) {
        std::error_code ec;
        const auto n = std::filesystem::file_size(u8p(path), ec);
        *size = ec ? -1 : (int64_t) n;
    }
    std::ifstream f(u8p(path), std::ios::binary);
    std::string line;
    if (!f || !std::getline(f, line)) {
        return std::string();
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

// a parsed state: the layers (v0.33.0's parser: the model name and the expert count must match) and the header's tokens
struct state_file {
    bool                   ok = false;
    state_meta             meta;
    std::vector<hot_layer> layers;
    std::string            header;
    int64_t                size  = -1; // -1: no file
    size_t                 n_bad = 0;  // broken layer lines left out
    std::string            why;
};

inline state_file read_state(const std::string & path, int64_t n_expert, const std::string & model) {
    state_file s;
    std::string text;
    if (!read_all(path, text)) {
        s.why = "no file";
        return s;
    }
    s.size = (int64_t) text.size();
    const size_t nl = text.find('\n');
    s.header = text.substr(0, nl == std::string::npos ? text.size() : nl);
    if (!s.header.empty() && s.header.back() == '\r') {
        s.header.pop_back();
    }
    s.meta = meta_parse(s.header);
    std::istringstream is(text);
    uint64_t steps = 0;
    s.ok = llama_moe_hotstate_parse(is, n_expert, model, s.layers, steps, s.why, &s.n_bad);
    return s;
}

// text to path through <path>.<pid>.tmp, flushed to the disk, then one rename over the old file (MoveFileEx with
// REPLACE_EXISTING on Windows: no moment without a file); the temporary file goes when anything fails
inline bool write_atomic(const std::string & path, const std::string & text, std::string & err) {
    namespace fs = std::filesystem;
#ifdef _WIN32
    const unsigned long pid = (unsigned long) GetCurrentProcessId();
#else
    const unsigned long pid = (unsigned long) getpid();
#endif
    const std::string tmp = path + "." + std::to_string(pid) + ".tmp";
#ifdef _WIN32
    FILE * f = nullptr;
    if (_wfopen_s(&f, u8p(tmp).c_str(), L"wb") != 0) {
        f = nullptr;
    }
#else
    FILE * f = fopen(tmp.c_str(), "wb");
#endif
    if (!f) {
        err = "cannot write " + tmp;
        return false;
    }
    bool ok = fwrite(text.data(), 1, text.size(), f) == text.size() && fflush(f) == 0;
#ifdef _WIN32
    ok = ok && _commit(_fileno(f)) == 0;
#else
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = fclose(f) == 0 && ok;
    if (!ok) {
        std::error_code ec;
        fs::remove(u8p(tmp), ec);
        err = "cannot write " + tmp;
        return false;
    }
#ifdef _WIN32
    if (!MoveFileExW(u8p(tmp).c_str(), u8p(path).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long e = GetLastError();
        std::error_code ec;
        fs::remove(u8p(tmp), ec);
        err = "cannot replace " + path + " (error " + std::to_string(e) + ")";
        return false;
    }
#else
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        std::error_code ec;
        fs::remove(u8p(tmp), ec);
        err = "cannot replace " + path;
        return false;
    }
#endif
    return true;
}

// the file as it is now becomes <path>.prev (the generation before this run's first save); false when there is no file
inline bool keep_prev(const std::string & path, std::string & err) {
    std::string text;
    if (!read_all(path, text)) {
        return false;
    }
    return write_atomic(path + ".prev", text, err);
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_STATEKEY] the identity of the weights file: the volume serial and the file index (Windows), the device and
// the inode (elsewhere); names that are hard links of one file have the same
struct file_id {
    bool     ok     = false;
    uint64_t vol    = 0;
    uint64_t idx    = 0;
    uint32_t nlinks = 0;
};

inline file_id file_identity(const std::string & path) {
    file_id id;
#ifdef _WIN32
    HANDLE h = CreateFileW(u8p(path).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return id;
    }
    BY_HANDLE_FILE_INFORMATION fi = {};
    if (GetFileInformationByHandle(h, &fi)) {
        id.ok     = true;
        id.vol    = fi.dwVolumeSerialNumber;
        id.idx    = ((uint64_t) fi.nFileIndexHigh << 32) | fi.nFileIndexLow;
        id.nlinks = fi.nNumberOfLinks;
    }
    CloseHandle(h);
#else
    struct stat st = {};
    if (stat(path.c_str(), &st) == 0) {
        id.ok     = true;
        id.vol    = (uint64_t) st.st_dev;
        id.idx    = (uint64_t) st.st_ino;
        id.nlinks = (uint32_t) st.st_nlink;
    }
#endif
    return id;
}

// the file names (no folder) of every hard link of the file, the given name first; elsewhere only the given name
inline std::vector<std::string> link_names(const std::string & path) {
    std::vector<std::string> out;
    const std::string self = u8p(path).filename().u8string();
    out.push_back(self);
#ifdef _WIN32
    std::vector<wchar_t> buf(1024);
    DWORD len = (DWORD) buf.size();
    HANDLE h = FindFirstFileNameW(u8p(path).c_str(), 0, &len, buf.data());
    if (h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_MORE_DATA) {
        buf.resize((size_t) len + 1);
        len = (DWORD) buf.size();
        h = FindFirstFileNameW(u8p(path).c_str(), 0, &len, buf.data());
    }
    if (h == INVALID_HANDLE_VALUE) {
        return out;
    }
    for (int guard = 0; guard < 1024; ++guard) {
        const std::string name = std::filesystem::path(std::wstring(buf.data())).filename().u8string();
        if (!name.empty() && std::find(out.begin(), out.end(), name) == out.end()) {
            out.push_back(name);
        }
        len = (DWORD) buf.size();
        if (!FindNextFileNameW(h, &len, buf.data())) {
            if (GetLastError() != ERROR_MORE_DATA) {
                break;
            }
            buf.resize((size_t) len + 1);
            len = (DWORD) buf.size();
            if (!FindNextFileNameW(h, &len, buf.data())) {
                break;
            }
        }
    }
    FindClose(h);
#endif
    return out;
}

// the key's file name: the model's name (letters, digits, '.', '_' and '-'; at most 64), the volume and the file index
inline std::string key_name(const std::string & model, const file_id & id) {
    std::string m;
    for (const char c : model) {
        const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                          c == '-';
        m += keep ? c : '_';
        if (m.size() >= 64) {
            break;
        }
    }
    if (m.empty()) {
        m = "model";
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "-%08llx-%016llx.hotstate", (unsigned long long) (id.vol & 0xffffffffull),
            (unsigned long long) id.idx);
    return m + buf;
}

// the key files in dir of the same model name as the key file key_path and another identity (the weights downloaded
// again, restored from a backup, copied or rebuilt under the same name get a new file index), as names for adopt_scan
// (without ".hotstate"); a key's .prev and temporary files are not keys
inline std::vector<std::string> key_peers(const std::string & dir, const std::string & key_path) {
    std::vector<std::string> out;
    const std::string self = u8p(key_path).filename().u8string();
    const size_t      tail = 1 + 8 + 1 + 16 + 9; // "-<volume>-<file index>.hotstate"
    auto is_hex = [](const std::string & s, size_t at, size_t n) {
        for (size_t i = at; i < at + n; ++i) {
            const char c = s[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                return false;
            }
        }
        return true;
    };
    auto is_key = [&](const std::string & s) {
        if (s.size() <= tail) {
            return false;
        }
        const size_t b = s.size() - tail;
        return s[b] == '-' && is_hex(s, b + 1, 8) && s[b + 9] == '-' && is_hex(s, b + 10, 16) &&
               s.compare(b + 26, 9, ".hotstate") == 0;
    };
    if (!is_key(self)) {
        return out;
    }
    const std::string model = self.substr(0, self.size() - tail);
    std::error_code ec;
    for (std::filesystem::directory_iterator it(u8p(dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().u8string();
        if (n != self && n.size() == self.size() && n.compare(0, model.size(), model) == 0 && is_key(n)) {
            out.push_back(n.substr(0, n.size() - 9));
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// the old state to adopt: of the candidate files that hold a usable state of this model, the one learned over the most
// decode steps (the first one on a tie); -1 when none
struct adopt_cand {
    std::string path;
    bool        ok      = false;
    uint64_t    learned = 0;
    size_t      n_res   = 0;
    std::string why;
};

inline int adopt_pick(const std::vector<adopt_cand> & c) {
    int best = -1;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].ok && (best < 0 || c[i].learned > c[(size_t) best].learned)) {
            best = (int) i;
        }
    }
    return best;
}

// reads the candidates <dir>/<name>.hotstate of every name; with layer_bytes (an expert's bytes by layer, 0: not known) a
// state with a layer of another expert size (another quantization of the model: the load skips such layers) is not usable
inline std::vector<adopt_cand> adopt_scan(const std::string & dir, const std::vector<std::string> & names, int64_t n_expert,
        const std::string & model, const std::vector<size_t> * layer_bytes = nullptr) {
    std::vector<adopt_cand> out;
    for (const auto & n : names) {
        adopt_cand c;
        c.path = (u8p(dir) / u8p(n + ".hotstate")).u8string();
        const state_file s = read_state(c.path, n_expert, model);
        c.ok      = s.ok;
        c.learned = s.meta.learned;
        c.why     = s.why;
        for (const auto & l : s.layers) {
            c.n_res += l.res.size();
            const size_t want = layer_bytes && l.il >= 0 && (size_t) l.il < layer_bytes->size() ? (*layer_bytes)[(size_t) l.il] : 0;
            if (c.ok && want != 0 && l.bytes != want) {
                c.ok  = false;
                c.why = "another expert size";
            }
        }
        if (s.why != "no file") {
            out.push_back(std::move(c));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// [TAG_FN_R41_PASSPRIO] the calling thread's page priority (0..7, normal 5): the pages it reads into RAM take it, and the
// system repurposes the standby pages of the lowest priority first. Through ntdll's ThreadPagePriority (class 24), as
// llama-fn-r41.h reads and raises the launcher's; no privilege is needed below normal. -1: not read (or not Windows)
inline int thread_page_prio() {
#ifdef _WIN32
    using nt_info_t = LONG (WINAPI *)(HANDLE, ULONG, PVOID, ULONG, PULONG);
    const HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    const nt_info_t q = nt ? (nt_info_t) (void *) GetProcAddress(nt, "NtQueryInformationThread") : nullptr;
    ULONG v = 0, n = 0;
    if (q && q(GetCurrentThread(), 24, &v, sizeof(v), &n) >= 0) {
        return (int) v;
    }
#endif
    return -1;
}

inline bool set_thread_page_prio(int prio) {
#ifdef _WIN32
    using nt_set_t = LONG (WINAPI *)(HANDLE, ULONG, PVOID, ULONG);
    const HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    const nt_set_t s = nt ? (nt_set_t) (void *) GetProcAddress(nt, "NtSetInformationThread") : nullptr;
    ULONG v = (ULONG) std::max(0, std::min(7, prio));
    return s && s(GetCurrentThread(), 24, &v, sizeof(v)) >= 0;
#else
    (void) prio;
    return false;
#endif
}

// the switch's value as a page priority: 0 (off) stays 0, else 1..4 (never normal or above: that is the default)
inline int passprio_value(int v) {
    return v <= 0 ? 0 : std::min(4, v);
}

// a long-lived thread follows a wanted page priority that can change while it runs: apply() before each piece of work
// sets it when it differs from the last one set (0: back to the thread's own value from before the first change)
struct page_prio_follow {
    int applied = 0;
    int orig    = -1;

    void apply(int want) {
        if (want == applied) {
            return;
        }
        if (orig < 0) {
            orig = thread_page_prio();
        }
        const int v = want > 0 ? want : orig;
        if (v >= 0) {
            (void) set_thread_page_prio(v);
        }
        applied = want;
    }
};

} // namespace fn_r41b
