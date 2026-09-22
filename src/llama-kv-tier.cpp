#include "llama-kv-tier.h"

#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-turbot-default-plan.h"   // [TAG_TURBOT_EMBED_PLAN] generated, checked in

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

//
// [TAG_TURBOT] plan path (llama-ext.h). Process-wide: common sets it before the fit probes create their contexts.
//

static std::mutex  g_turbot_plan_mutex;
static std::string g_turbot_plan_path;

void llama_turbot_set_plan_path(const char * path) {
    std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
    g_turbot_plan_path = path ? path : "";
}

std::string llama_turbot_get_plan_path() {
    std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
    return g_turbot_plan_path;
}

// [TAG_TURBOT_ANY_SIDECAR] sidecar path (llama-ext.h), set by common when <model>.turbot.plan exists
static std::string g_turbot_sidecar_path;

// tests only (llama_turbot_test_set_model_fingerprint)
static std::string g_turbot_test_fingerprint;

void llama_turbot_set_sidecar_path(const char * path) {
    std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
    g_turbot_sidecar_path = path ? path : "";
}

std::string llama_turbot_get_sidecar_path() {
    std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
    return g_turbot_sidecar_path;
}

void llama_turbot_test_set_model_fingerprint(const char * fingerprint) {
    std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
    g_turbot_test_fingerprint = fingerprint ? fingerprint : "";
}

//
// [TAG_TURBOT_ANY_PLAN] env switches, validated geometries, budget type
//

// true when env name is set to exactly value
static bool llama_turbot_env_is(const char * name, const char * value) {
    const char * e = getenv(name);
    return e != nullptr && strcmp(e, value) == 0;
}

llama_turbot_switches llama_turbot_read_switches() {
    llama_turbot_switches sw;

    sw.any = !llama_turbot_env_is("LLAMA_TURBOT_ANY", "0");
    if (!sw.any) {
        sw.auto_plan           = false;
        sw.auto_all            = false;
        sw.auto_budget_turbo5p = false;
        sw.sidecar             = false;
        sw.iswa                = false;
        sw.multi_stream        = false;
        sw.swa_type.clear();
        return sw;
    }

    sw.auto_plan           = !llama_turbot_env_is("LLAMA_TURBOT_AUTO_PLAN", "0");
    sw.auto_all            =  llama_turbot_env_is("LLAMA_TURBOT_AUTO_PLAN", "all");
    sw.auto_budget_turbo5p =  llama_turbot_env_is("LLAMA_TURBOT_AUTO_BUDGET", "turbo5p");
    sw.sidecar             = !llama_turbot_env_is("LLAMA_TURBOT_SIDECAR", "0");
    sw.iswa                = !llama_turbot_env_is("LLAMA_TURBOT_ISWA", "0");
    sw.multi_stream        = !llama_turbot_env_is("LLAMA_TURBOT_MULTI_STREAM", "0");

    const char * swa_type = getenv("LLAMA_TURBOT_SWA_TYPE");
    sw.swa_type = swa_type ? swa_type : "";

    return sw;
}

bool llama_turbot_auto_geom_validated(int head_dim, int n_head_kv) {
    // head dim x KV heads that an automatic plan was validated on (turbot_guard.py G5 runs)
    static constexpr int validated[][2] = {
        { 256, 4 },
        { 256, 2 },
    };
    for (const auto & v : validated) {
        if (v[0] == head_dim && v[1] == n_head_kv) {
            return true;
        }
    }
    return false;
}

ggml_type llama_turbot_budget_type(uint32_t row_elems) {
    if (row_elems % 1024 == 0) {
        return GGML_TYPE_TURBO5P_0;
    }
    if (row_elems % 512 == 0) {
        return GGML_TYPE_TURBO5P512_0;
    }
    return GGML_TYPE_TURBO4_0;
}

// the name of a budget type in plan text and log lines (tools/turbot/turbot_plan.py BUDGET_TYPES)
static const char * llama_turbot_budget_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_TURBO5P_0:    return "turbo5p";
        case GGML_TYPE_TURBO5P512_0: return "turbo5p512";
        case GGML_TYPE_TURBO4_0:     return "turbo4";
        default:                     return ggml_type_name(t);
    }
}

//
// [TAG_TURBOT_ANY_GEOM] the geometry of each layer of a cache shape
//

struct llama_turbot_shape_geom {
    int      flags     = 0;
    int      nr        = GGML_TURBOT_N_HEAD;
    uint16_t head_dim  = GGML_TURBOT_HEAD_DIM;
    uint8_t  n_head_kv = GGML_TURBOT_N_HEAD;
};

// il -> geometry, the first entry of a repeated il wins. false and msg when turbot has no layout for a layer, or, with
// LLAMA_TURBOT_ANY=0, for anything but 4 KV heads x 256.
static bool llama_turbot_shape_geoms(const llama_turbot_cache_shape & shape, bool any, std::map<int32_t, llama_turbot_shape_geom> & out,
        std::string & msg) {
    out.clear();
    for (const auto & lg : shape.layers) {
        if (out.count(lg.il)) {
            continue;
        }
        const int flags = ggml_turbot_geom_flags((int) lg.head_dim, (int) lg.n_head_kv);
        if (flags < 0 || (!any && flags != 0)) {
            msg = format("attention layer %d: turbot has no layout for %u KV heads x %u%s", lg.il, (unsigned) lg.n_head_kv,
                    (unsigned) lg.head_dim, any ? "" : " (LLAMA_TURBOT_ANY=0: only 4 KV heads x 256)");
            return false;
        }
        llama_turbot_shape_geom g;
        g.flags     = flags;
        g.nr        = ggml_turbot_geom_nr((unsigned) flags);
        g.head_dim  = lg.head_dim;
        g.n_head_kv = lg.n_head_kv;
        out[lg.il]  = g;
    }
    return true;
}

// the shape of the old overloads: 4 KV heads x 256 on every layer, one stream
static llama_turbot_cache_shape llama_turbot_shape_4x256(const std::vector<int32_t> & attn_layers, uint32_t kv_size) {
    llama_turbot_cache_shape shape;
    shape.layers.reserve(attn_layers.size());
    for (const int32_t il : attn_layers) {
        shape.layers.push_back({ il, (uint16_t) GGML_TURBOT_HEAD_DIM, (uint8_t) GGML_TURBOT_N_HEAD });
    }
    shape.kv_size   = kv_size;
    shape.n_stream  = 1;
    shape.n_seq_max = 1;
    return shape;
}

// cells of all streams
static uint64_t llama_turbot_shape_cells(const llama_turbot_cache_shape & shape) {
    return (uint64_t) shape.kv_size * std::max<uint32_t>(1, shape.n_stream);
}

//
// [TAG_TURBOT] plan file (SPEC 9.1)
//

static bool llama_turbot_parse_int(const std::string & s, int64_t & out) {
    if (s.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long long v = strtoll(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0') {
        return false;
    }
    out = (int64_t) v;
    return true;
}

static int llama_turbot_popcount(uint64_t x) {
    int n = 0;
    while (x) {
        x &= x - 1;
        ++n;
    }
    return n;
}

// "<w0> <w1> ... <w(nr-1)>" of the malformed-line message
static std::string llama_turbot_widths_form(int nr) {
    std::string s;
    for (int r = 0; r < nr; ++r) {
        s += format("%s<w%d>", r ? " " : "", r);
    }
    return s;
}

// quiet: no warning lines ([TAG_TURBOT_EMBED_PLAN] llama_turbot_plan_matches is pure)
// [TAG_TURBOT_ANY_PLAN] against a shape: a layer with nr runs has nr widths per side on its L and Y lines. With 4 KV
// heads x 256 on every layer the accepted plans, the errors and the hash are the ones of the attn_layers parser.
static bool llama_turbot_plan_parse_impl(const std::string & text, const std::string & source,
        const llama_turbot_cache_shape & shape, llama_turbot_plan & plan, std::string & err,
        bool quiet = false) {
    const auto fail = [&](int line_no, const std::string & msg) {
        err = line_no > 0 ? format("turbot: plan %s line %d: %s", source.c_str(), line_no, msg.c_str())
                          : format("turbot: plan %s: %s", source.c_str(), msg.c_str());
        return false;
    };

    // [TAG_TURBOT_ANY_GEOM] runs of every layer of the cache
    std::map<int32_t, llama_turbot_shape_geom> geoms;
    {
        std::string msg;
        if (!llama_turbot_shape_geoms(shape, llama_turbot_read_switches().any, geoms, msg)) {
            return fail(0, msg);
        }
    }

    // one L or Y line: the K widths and the V widths of a layer, one per run (runs r >= nr stay 0)
    struct widths_line {
        int     line_no = 0;
        uint8_t k[4]    = {};
        uint8_t v[4]    = {};
    };

    std::map<int32_t, widths_line> lines_l;
    std::map<int32_t, widths_line> lines_y;

    int     line_pool = 0;
    int     line_cap  = 0;
    int64_t pool      = GGML_TURBOT_POOL_DEFAULT;
    int64_t cap       = GGML_TURBOT_CAP_DEFAULT;

    int line_bench = 0;

    std::istringstream in(text);
    std::string        line;
    int                line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;

        const size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line.resize(hash_pos);
        }

        // whitespace includes the '\r' of a CRLF file
        std::istringstream       ss(line);
        std::vector<std::string> tok;
        for (std::string t; ss >> t; ) {
            tok.push_back(t);
        }
        if (tok.empty()) {
            continue;
        }

        const std::string & tag = tok[0];

        if (tag == "L" || tag == "Y") {
            const bool is_l = tag == "L";

            // [TAG_TURBOT_ANY_GEOM] the runs of the layer the line names; 4 (the 12-token form, whose errors read as
            // before) when tok[1] is not one of the cache's layers
            int         nr   = GGML_TURBOT_N_HEAD;
            const char * run = "head";
            {
                int64_t il_peek = 0;
                if (tok.size() >= 2 && llama_turbot_parse_int(tok[1], il_peek) && il_peek >= 0 &&
                        il_peek <= std::numeric_limits<int32_t>::max()) {
                    const auto itg = geoms.find((int32_t) il_peek);
                    if (itg != geoms.end()) {
                        nr  = itg->second.nr;
                        run = (itg->second.flags & GGML_TURBOT_GEOM_D128) ? "run" : "head";
                    }
                }
            }

            if (tok.size() != (size_t) (4 + 2*nr) || tok[2] != "K" || tok[3 + nr] != "V") {
                const std::string w = llama_turbot_widths_form(nr);
                return fail(line_no, format("malformed %s line, expected '%s <il> K %s V %s'",
                        tag.c_str(), tag.c_str(), w.c_str(), w.c_str()));
            }

            int64_t il = 0;
            if (!llama_turbot_parse_int(tok[1], il) || il < 0 || il > std::numeric_limits<int32_t>::max()) {
                return fail(line_no, "bad layer index '" + tok[1] + "'");
            }

            widths_line wl;
            wl.line_no = line_no;
            for (int h = 0; h < nr; ++h) {
                int64_t wk = 0;
                int64_t wv = 0;
                if (!llama_turbot_parse_int(tok[3 + h], wk) || !llama_turbot_parse_int(tok[4 + nr + h], wv)) {
                    return fail(line_no, format("malformed %s line: widths must be integers", tag.c_str()));
                }
                if (is_l) {
                    if (wk < GGML_TURBOT_B_MIN || wk > GGML_TURBOT_B_MAX || wv < GGML_TURBOT_B_MIN || wv > GGML_TURBOT_B_MAX) {
                        return fail(line_no, format("layer %d %s %d: old width must be in [%d, %d] (K %lld, V %lld)",
                                (int) il, run, h, GGML_TURBOT_B_MIN, GGML_TURBOT_B_MAX, (long long) wk, (long long) wv));
                    }
                } else {
                    // the lower bound b+1 depends on the L line and is checked once every line is read
                    if (wk < 1 || wk > GGML_TURBOT_Y_MAX || wv < 1 || wv > GGML_TURBOT_Y_MAX) {
                        return fail(line_no, format("layer %d %s %d: young width must be in [b+1, %d] (K %lld, V %lld)",
                                (int) il, run, h, GGML_TURBOT_Y_MAX, (long long) wk, (long long) wv));
                    }
                }
                wl.k[h] = (uint8_t) wk;
                wl.v[h] = (uint8_t) wv;
            }

            auto & dst = is_l ? lines_l : lines_y;
            const auto it = dst.find((int32_t) il);
            if (it != dst.end()) {
                return fail(line_no, format("duplicate %s line for layer %d (first on line %d)", tag.c_str(), (int) il, it->second.line_no));
            }
            dst[(int32_t) il] = wl;
        } else if (tag == "POOL" || tag == "CAP") {
            const bool is_pool = tag == "POOL";
            int64_t    v       = 0;
            if (tok.size() != 2 || !llama_turbot_parse_int(tok[1], v)) {
                return fail(line_no, format("malformed %s line, expected '%s <cells>'", tag.c_str(), tag.c_str()));
            }
            if (is_pool) {
                if (line_pool) {
                    return fail(line_no, format("duplicate POOL line (first on line %d)", line_pool));
                }
                if (v < 0 || v % GGML_TURBOT_GRANULE != 0) {
                    return fail(line_no, format("POOL %lld must be a non-negative multiple of %d", (long long) v, GGML_TURBOT_GRANULE));
                }
                line_pool = line_no;
                pool      = v;
            } else {
                if (line_cap) {
                    return fail(line_no, format("duplicate CAP line (first on line %d)", line_cap));
                }
                if (v < 0) {
                    return fail(line_no, format("CAP %lld must not be negative", (long long) v));
                }
                if (v > (int64_t) std::numeric_limits<uint32_t>::max()) {
                    return fail(line_no, format("CAP %lld is too large", (long long) v));
                }
                line_cap = line_no;
                cap      = v;
            }
        } else if (tag == "W" || tag == "W2" || tag == "M") {
            // kvfq bench keys: the bench plans carry them, turbot has no use for them
            if (!line_bench) {
                line_bench = line_no;
            }
        } else {
            return fail(line_no, "unknown tag '" + tag + "'");
        }
    }

    // checked on the resolved value (explicit POOL or the implicit default): a pool larger than the cache is clamped to
    // the cache, rounded down to whole granules, so small contexts (llama-bench at low depth, perplexity at 32K, fit
    // probes) run with every granule able to be young instead of being refused. The 262144-cell server is unaffected.
    // [TAG_TURBOT_ANY_STREAMS] the cache is kv_size cells per stream: the pool serves all streams
    const uint64_t kv_cells = llama_turbot_shape_cells(shape);
    if (pool > (int64_t) kv_cells) {
        const int64_t clamped = ((int64_t) kv_cells / GGML_TURBOT_GRANULE) * GGML_TURBOT_GRANULE;
        if (!quiet) {
            LLAMA_LOG_WARN("%s: turbot plan %s%s: POOL %lld is larger than the cache (%llu cells), using %lld\n", __func__, source.c_str(),
                    line_pool ? format(" line %d", line_pool).c_str() : "", (long long) pool, (unsigned long long) kv_cells, (long long) clamped);
        }
        pool = clamped;
    }

    std::set<int32_t> attn;
    for (const auto & it : geoms) {
        attn.insert(it.first);
    }

    for (const auto & [il, wl] : lines_l) {
        if (attn.count(il) == 0) {
            return fail(wl.line_no, format("layer %d is not an attention layer of this cache", il));
        }
    }
    for (const auto & [il, wl] : lines_y) {
        if (attn.count(il) == 0) {
            return fail(wl.line_no, format("layer %d is not an attention layer of this cache", il));
        }
    }

    llama_turbot_plan res;
    res.path = plan.path;

    for (const int32_t il : attn) {
        const auto itl = lines_l.find(il);
        if (itl == lines_l.end()) {
            return fail(0, format("missing L line for attention layer %d", il));
        }
        const widths_line & wl  = itl->second;
        const auto          ity = lines_y.find(il);

        // [TAG_TURBOT_ANY_GEOM] nr runs; the widths of runs r >= nr stay 0
        const llama_turbot_shape_geom & geom = geoms.at(il);
        const char * run = (geom.flags & GGML_TURBOT_GEOM_D128) ? "run" : "head";

        uint8_t yk[4] = {};
        uint8_t yv[4] = {};
        for (int h = 0; h < geom.nr; ++h) {
            yk[h] = ity != lines_y.end() ? ity->second.k[h] : (uint8_t) GGML_TURBOT_Y_DEFAULT;
            yv[h] = ity != lines_y.end() ? ity->second.v[h] : (uint8_t) GGML_TURBOT_Y_DEFAULT;
            if (yk[h] <= wl.k[h] || yk[h] > GGML_TURBOT_Y_MAX || yv[h] <= wl.v[h] || yv[h] > GGML_TURBOT_Y_MAX) {
                return fail(ity != lines_y.end() ? ity->second.line_no : wl.line_no,
                        format("layer %d %s %d: young width must be in [b+1, %d] (K b %d y %d, V b %d y %d)",
                            il, run, h, GGML_TURBOT_Y_MAX, wl.k[h], yk[h], wl.v[h], yv[h]));
            }
        }

        ggml_turbot_layer l;
        if (!ggml_turbot_layer_init_geom(&l, wl.k, wl.v, yk, yv, (unsigned) geom.flags)) {
            return fail(wl.line_no, format("layer %d: illegal widths", il));
        }
        res.layers[il] = l;
    }

    if (res.layers.empty()) {
        return fail(0, "the cache holds no attention layers");
    }

    res.pool_cells = (uint32_t) pool;
    res.cap_cells  = (uint32_t) cap;

    uint64_t h = GGML_TURBOT_FNV_OFFSET;
    for (const auto & [il, l] : res.layers) {
        h = ggml_turbot_plan_hash_layer(h, il, &l);
    }
    res.hash = ggml_turbot_plan_hash_finish(h, res.pool_cells, res.cap_cells);

    if (line_bench && !quiet) {
        LLAMA_LOG_WARN("%s: turbot plan %s line %d: the kvfq bench keys W, W2 and M are ignored\n", __func__, source.c_str(), line_bench);
    }

    plan = std::move(res);
    err.clear();

    return true;
}

bool llama_turbot_plan_parse_text(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err) {
    return llama_turbot_plan_parse_impl(text, "<text>", llama_turbot_shape_4x256(attn_layers, kv_size), plan, err);
}

bool llama_turbot_plan_parse_shape(const std::string & text, const std::string & source, const llama_turbot_cache_shape & shape,
                                   llama_turbot_plan & plan, std::string & err, bool quiet) {
    return llama_turbot_plan_parse_impl(text, source, shape, plan, err, quiet);
}

static bool llama_turbot_plan_read_file(const std::string & path, std::string & text, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = format("turbot: cannot open plan file %s", path.c_str());
        return false;
    }

    std::stringstream buf;
    buf << f.rdbuf();
    text = buf.str();

    return true;
}

bool llama_turbot_plan_parse_file(const std::string & path, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err) {
    std::string text;
    if (!llama_turbot_plan_read_file(path, text, err)) {
        return false;
    }

    if (!llama_turbot_plan_parse_impl(text, path, llama_turbot_shape_4x256(attn_layers, kv_size), plan, err)) {
        return false;
    }

    plan.path = path;

    return true;
}

//
// [TAG_TURBOT_EMBED_PLAN] built-in default plan (src/llama-turbot-default-plan.h) and plan source
//

const char * llama_turbot_default_plan_text() {
    return LLAMA_TURBOT_DEFAULT_PLAN_TEXT;
}

uint64_t llama_turbot_default_plan_hash() {
    return LLAMA_TURBOT_DEFAULT_PLAN_HASH;
}

bool llama_turbot_plan_matches(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                               std::string & why, const std::string & source) {
    llama_turbot_plan plan;
    return llama_turbot_plan_parse_impl(text, source, llama_turbot_shape_4x256(attn_layers, kv_size), plan, why, /*quiet =*/ true);
}

llama_turbot_plan_source llama_turbot_plan_get_source() {
    std::string value  = llama_turbot_get_plan_path();
    std::string set_by = "--kv-tier-plan";
    if (value.empty()) {
        const char * LLAMA_TURBOT_PLAN = getenv("LLAMA_TURBOT_PLAN");
        value  = LLAMA_TURBOT_PLAN ? LLAMA_TURBOT_PLAN : "";
        set_by = "LLAMA_TURBOT_PLAN";
    }

    llama_turbot_plan_source src;
    if (value.empty()) {
        src.origin = LLAMA_TURBOT_PLAN_BUILTIN_AUTO;
        src.name   = LLAMA_TURBOT_PLAN_BUILTIN_NAME;
    } else if (value == LLAMA_TURBOT_PLAN_KEYWORD_DEFAULT) {
        src.origin = LLAMA_TURBOT_PLAN_BUILTIN_FORCED;
        src.name   = LLAMA_TURBOT_PLAN_BUILTIN_NAME;
        src.set_by = set_by;
    } else if (value == LLAMA_TURBOT_PLAN_KEYWORD_AUTO && llama_turbot_read_switches().any) {
        // [TAG_TURBOT_ANY_PLAN] the automatic plan of the cache shape; with LLAMA_TURBOT_ANY=0 "auto" names a file, as before
        src.origin = LLAMA_TURBOT_PLAN_AUTO_FORCED;
        src.name   = LLAMA_TURBOT_PLAN_AUTO_NAME;
        src.set_by = set_by;
    } else {
        src.origin = LLAMA_TURBOT_PLAN_FILE;
        src.path   = value;
        src.name   = value;
        src.set_by = set_by;
    }

    return src;
}

llama_turbot_plan_source llama_turbot_plan_sidecar_source() {
    llama_turbot_plan_source src;
    src.origin = LLAMA_TURBOT_PLAN_BUILTIN_AUTO;
    src.name   = LLAMA_TURBOT_PLAN_BUILTIN_NAME;

    if (!llama_turbot_read_switches().sidecar) {
        return src;
    }

    const std::string path = llama_turbot_get_sidecar_path();
    if (path.empty()) {
        return src;
    }

    src.origin = LLAMA_TURBOT_PLAN_SIDECAR;
    src.path   = path;
    src.name   = path;

    return src;
}

bool llama_turbot_plan_read(const llama_turbot_plan_source & src, std::string & text, std::string & err) {
    if (src.origin == LLAMA_TURBOT_PLAN_FILE || src.origin == LLAMA_TURBOT_PLAN_SIDECAR) {
        return llama_turbot_plan_read_file(src.path, text, err);
    }

    if (src.origin == LLAMA_TURBOT_PLAN_AUTO_FORCED) {
        text.clear();
        err = format("turbot: plan %s: an automatic plan is generated from the cache shape (llama_turbot_plan_auto_text)",
                LLAMA_TURBOT_PLAN_AUTO_NAME);
        return false;
    }

    text = LLAMA_TURBOT_DEFAULT_PLAN_TEXT;
    err.clear();

    return true;
}

//
// [TAG_TURBOT_ANY_PLAN] automatic plan (tools/turbot/turbot_plan.py auto_plan mirrors this, text and hash)
//

#define LLAMA_TURBOT_AUTO_POOL_MIN_SEQ 1024   // young pool floor per sequence, plus the quota slack

static const double LLAMA_TURBOT_MIB = 1024.0*1024.0;

// the bytes an automatic plan must fit: the resolver's turbot fallback for every layer-side of every cell
struct llama_turbot_budget {
    ggml_type   fallback = GGML_TYPE_TURBO5P_0;   // turbo5p if every row is a multiple of 1024 values, turbo5p512 if of
                                                  // 512, else turbo4 (one type for the cache, as the resolver picks it)
    std::string name;                             // the fallback, or "turbo5p" with LLAMA_TURBOT_AUTO_BUDGET=turbo5p
    uint64_t    bytes    = 0;
};

static llama_turbot_budget llama_turbot_budget_of(const std::map<int32_t, llama_turbot_shape_geom> & geoms, uint64_t kv_cells,
        bool rate_turbo5p) {
    bool all_1024 = true;
    bool all_512  = true;
    for (const auto & it : geoms) {
        const uint32_t row = (uint32_t) ggml_turbot_geom_row_elems((unsigned) it.second.flags);
        all_1024 = all_1024 && row % 1024 == 0;
        all_512  = all_512  && row % 512  == 0;
    }

    llama_turbot_budget b;
    b.fallback = llama_turbot_budget_type(all_1024 ? 1024 : all_512 ? 512 : 256);
    b.name     = rate_turbo5p ? "turbo5p" : llama_turbot_budget_name(b.fallback);

    // turbo5p rate: 656 B per 1024 values, exact for every row of whole 256-value runs
    const uint64_t t5p_row = (uint64_t) ggml_row_size(GGML_TYPE_TURBO5P_0, 1024);

    uint64_t per_cell = 0;
    for (const auto & it : geoms) {
        const uint64_t row = (uint64_t) ggml_turbot_geom_row_elems((unsigned) it.second.flags);
        per_cell += 2*(rate_turbo5p ? row*t5p_row/1024 : (uint64_t) ggml_row_size(b.fallback, (int64_t) row));
    }
    b.bytes = kv_cells*per_cell;

    return b;
}

// base rows of every cell plus the young pool of a parsed plan
static uint64_t llama_turbot_plan_bytes(const llama_turbot_plan & plan, uint64_t kv_cells) {
    uint64_t base  = 0;
    uint64_t young = 0;
    for (const auto & it : plan.layers) {
        base  += (uint64_t) it.second.k.base_row_bytes + it.second.v.base_row_bytes;
        young += it.second.pool_row_bytes;
    }
    return kv_cells*base + (uint64_t) plan.pool_cells*young;
}

// mean old width over the runs of both sides
static double llama_turbot_plan_mean_width(const llama_turbot_plan & plan) {
    uint64_t sum  = 0;
    uint64_t runs = 0;
    for (const auto & it : plan.layers) {
        sum  += (uint64_t) it.second.k.s  + it.second.v.s;
        runs += (uint64_t) it.second.k.nr + it.second.v.nr;
    }
    return runs ? (double) sum / (double) runs : 0.0;
}

static bool llama_turbot_auto_impl(const llama_turbot_cache_shape & shape, const llama_turbot_switches & sw, std::string & text,
        std::string & why) {
    std::map<int32_t, llama_turbot_shape_geom> geoms;
    if (!llama_turbot_shape_geoms(shape, sw.any, geoms, why)) {
        return false;
    }
    if (geoms.empty()) {
        why = "the cache holds no attention layers";
        return false;
    }
    if (shape.kv_size == 0 || shape.kv_size % GGML_TURBOT_GRANULE != 0) {
        why = format("kv_size %u is not a positive multiple of %d", shape.kv_size, GGML_TURBOT_GRANULE);
        return false;
    }

    const uint64_t n_stream = std::max<uint32_t>(1, shape.n_stream);
    const uint64_t n_seq    = std::max<uint32_t>(1, shape.n_seq_max);
    const uint64_t kv_cells = llama_turbot_shape_cells(shape);
    const uint64_t gran     = GGML_TURBOT_GRANULE;
    const uint64_t step     = gran*n_stream;                                     // POOL_s = POOL / n_stream whole granules
    const uint64_t cap      = GGML_TURBOT_CAP_DEFAULT;
    const uint64_t slack    = gran*GGML_TURBOT_QUOTA_SLACK_GRANULES;
    const int      y        = GGML_TURBOT_Y_DEFAULT;

    const llama_turbot_budget budget = llama_turbot_budget_of(geoms, kv_cells, sw.auto_budget_turbo5p);

    int nr_max = 0;
    for (const auto & it : geoms) {
        nr_max = std::max(nr_max, it.second.nr);
    }

    // K widths of layer nr runs with m runs at 5 bits: K takes ceil(m/2), V floor(m/2), lowest run first
    const auto widths = [](int m, int nr, int side, uint8_t * b) {
        const int n5 = std::min(side == 0 ? (m + 1)/2 : m/2, nr);
        for (int r = 0; r < nr; ++r) {
            b[r] = (uint8_t) (r < n5 ? 5 : 4);
        }
    };

    // bytes per cell of the base rows and per pool row of the young parts
    const auto per_cell = [&](int m, uint64_t & base, uint64_t & young) {
        base  = 0;
        young = 0;
        for (const auto & it : geoms) {
            const int nr = it.second.nr;
            for (int side = 0; side < 2; ++side) {
                uint8_t b[GGML_TURBOT_MAX_RUNS] = {};
                widths(m, nr, side, b);
                uint64_t s = 0;
                uint64_t r = 0;
                for (int run = 0; run < nr; ++run) {
                    s += b[run];
                    r += (uint64_t) (y - b[run]);
                }
                base  += 32*s + 16;
                young += 32*r + 16;
            }
        }
    };

    uint64_t pool = std::min((n_seq*(cap + slack) + gran - 1)/gran*gran, kv_cells/gran*gran);
    pool = pool/step*step;

    int m_best = -1;
    for (int m = 2*nr_max; m >= 0; --m) {
        uint64_t base  = 0;
        uint64_t young = 0;
        per_cell(m, base, young);
        if (kv_cells*base + pool*young <= budget.bytes) {
            m_best = m;
            break;
        }
    }

    if (m_best < 0) {
        // even 4 bits everywhere does not fit this pool: shrink it
        uint64_t base  = 0;
        uint64_t young = 0;
        per_cell(0, base, young);
        if (kv_cells*base > budget.bytes) {
            why = format("4-bit old rows alone need %.2f MiB, above the %.2f MiB of %s", kv_cells*base/LLAMA_TURBOT_MIB,
                    budget.bytes/LLAMA_TURBOT_MIB, budget.name.c_str());
            return false;
        }
        pool   = (budget.bytes - kv_cells*base)/young;
        pool   = pool/step*step;
        m_best = 0;
    }

    const uint64_t pool_min = n_seq*(LLAMA_TURBOT_AUTO_POOL_MIN_SEQ + slack);
    if (pool < pool_min) {
        why = format("the young pool would be %llu cells in the %.2f MiB of %s, below %llu (%llu sequences x %d)",
                (unsigned long long) pool, budget.bytes/LLAMA_TURBOT_MIB, budget.name.c_str(), (unsigned long long) pool_min,
                (unsigned long long) n_seq, (int) (LLAMA_TURBOT_AUTO_POOL_MIN_SEQ + slack));
        return false;
    }

    uint64_t base  = 0;
    uint64_t young = 0;
    per_cell(m_best, base, young);
    const uint64_t bytes = kv_cells*base + pool*young;

    uint64_t sum_w  = 0;
    uint64_t n_runs = 0;
    for (const auto & it : geoms) {
        for (int side = 0; side < 2; ++side) {
            uint8_t b[GGML_TURBOT_MAX_RUNS] = {};
            widths(m_best, it.second.nr, side, b);
            for (int r = 0; r < it.second.nr; ++r) {
                sum_w += b[r];
            }
            n_runs += (uint64_t) it.second.nr;
        }
    }

    // the geometries in order of first appearance by il, "<count> x <KV heads>x<head dim>"
    std::vector<std::pair<std::pair<int, int>, int>> groups;
    for (const auto & it : geoms) {
        const std::pair<int, int> key((int) it.second.n_head_kv, (int) it.second.head_dim);
        auto g = std::find_if(groups.begin(), groups.end(), [&](const auto & e) { return e.first == key; });
        if (g == groups.end()) {
            groups.push_back({ key, 1 });
        } else {
            g->second++;
        }
    }
    std::string geo;
    for (size_t i = 0; i < groups.size(); ++i) {
        geo += format("%s%d x %dx%d", i ? ", " : "", groups[i].second, groups[i].first.first, groups[i].first.second);
    }

    text  = "# turbot auto plan v1\n";
    text += format("# shape: %d attention layers: %s (KV heads x head dim); kv_size %u, n_stream %u, n_seq_max %u\n",
            (int) geoms.size(), geo.c_str(), shape.kv_size, (unsigned) n_stream, (unsigned) n_seq);
    if (sw.auto_budget_turbo5p) {
        text += format("# budget: turbo5p rate, 656 B per 1024 values (LLAMA_TURBOT_AUTO_BUDGET=turbo5p); the fallback type is %s\n",
                llama_turbot_budget_name(budget.fallback));
    } else {
        text += format("# budget: the bytes of the fallback type %s\n", budget.name.c_str());
    }
    text += format("# size: %.2f MiB (%s %.2f MiB); old widths 4/5 mean %.3f, young %d, uncalibrated\n",
            bytes/LLAMA_TURBOT_MIB, budget.name.c_str(), budget.bytes/LLAMA_TURBOT_MIB, (double) sum_w/(double) n_runs, y);
    for (const auto & it : geoms) {
        const int nr = it.second.nr;
        uint8_t bk[GGML_TURBOT_MAX_RUNS] = {};
        uint8_t bv[GGML_TURBOT_MAX_RUNS] = {};
        widths(m_best, nr, 0, bk);
        widths(m_best, nr, 1, bv);
        std::string line = format("L %d K", it.first);
        for (int r = 0; r < nr; ++r) {
            line += format(" %d", bk[r]);
        }
        line += " V";
        for (int r = 0; r < nr; ++r) {
            line += format(" %d", bv[r]);
        }
        text += line + "\n";
    }
    text += format("POOL %llu\n", (unsigned long long) pool);
    text += format("CAP %llu\n",  (unsigned long long) cap);

    why.clear();

    return true;
}

bool llama_turbot_plan_auto_text(const llama_turbot_cache_shape & shape, std::string & text, std::string & why) {
    return llama_turbot_auto_impl(shape, llama_turbot_read_switches(), text, why);
}

// may step 4 give shape an automatic plan? (the validated geometries, LLAMA_TURBOT_AUTO_PLAN=all for every supported
// one, LLAMA_TURBOT_AUTO_BUDGET=turbo5p for the one-run geometries)
static bool llama_turbot_auto_allowed(const llama_turbot_cache_shape & shape, const llama_turbot_switches & sw, std::string & why) {
    std::map<int32_t, llama_turbot_shape_geom> geoms;
    if (!llama_turbot_shape_geoms(shape, sw.any, geoms, why)) {
        return false;
    }
    for (const auto & it : geoms) {
        const auto & g = it.second;
        if (llama_turbot_auto_geom_validated(g.head_dim, g.n_head_kv) || sw.auto_all || (sw.auto_budget_turbo5p && g.nr == 1)) {
            continue;
        }
        why = format("layer %d: %u KV heads x %u is not validated for an automatic plan (LLAMA_TURBOT_AUTO_PLAN=all%s takes it)",
                it.first, (unsigned) g.n_head_kv, (unsigned) g.head_dim, g.nr == 1 ? " or LLAMA_TURBOT_AUTO_BUDGET=turbo5p" : "");
        return false;
    }
    return true;
}

//
// [TAG_TURBOT_ANY_SIDECAR] model fingerprint and the sidecar check
//

static std::string llama_turbot_trim(const std::string & s) {
    const char * ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::string llama_turbot_fingerprint_text(const std::string & arch, const std::string & basename, const std::string & size,
        const llama_turbot_cache_shape & shape) {
    std::map<int32_t, std::pair<int, int>> layers;   // il -> head dim, KV heads; the first entry of a repeated il wins
    for (const auto & lg : shape.layers) {
        layers.emplace(lg.il, std::make_pair((int) lg.head_dim, (int) lg.n_head_kv));
    }

    std::string ils;
    std::string geo_all;
    bool        uniform = true;
    for (const auto & it : layers) {
        ils     += format("%s%d", ils.empty() ? "" : ",", it.first);
        geo_all += format("%s%dx%d", geo_all.empty() ? "" : ",", it.second.first, it.second.second);
        uniform  = uniform && it.second == layers.begin()->second;
    }
    const std::string geo = layers.empty() ? "" :
        uniform ? format("%dx%d", layers.begin()->second.first, layers.begin()->second.second) : geo_all;

    return format("arch=%s basename=%s size=%s layers=%s geom=%s", arch.c_str(), basename.c_str(), size.c_str(), ils.c_str(), geo.c_str());
}

// one metadata value of the model, "" when the key is missing
static std::string llama_turbot_meta(const llama_model * model, const char * key) {
    std::vector<char> buf(256);
    int32_t n = llama_model_meta_val_str(model, key, buf.data(), buf.size());
    if (n < 0) {
        return "";
    }
    if ((size_t) n >= buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_model_meta_val_str(model, key, buf.data(), buf.size());
        if (n < 0) {
            return "";
        }
    }
    return std::string(buf.data(), std::min((size_t) n, buf.size() - 1));
}

std::string llama_turbot_model_fingerprint(const llama_turbot_cache_shape & shape) {
    if (shape.model == nullptr) {
        return "";
    }
    {
        std::lock_guard<std::mutex> lock(g_turbot_plan_mutex);
        if (!g_turbot_test_fingerprint.empty()) {
            return g_turbot_test_fingerprint;
        }
    }
    return llama_turbot_fingerprint_text(
            llama_turbot_meta(shape.model, "general.architecture"),
            llama_turbot_meta(shape.model, "general.basename"),
            llama_turbot_meta(shape.model, "general.size_label"),
            shape);
}

bool llama_turbot_sidecar_accepts(const std::string & text, const std::string & fingerprint, std::string & why) {
    bool        verified  = false;
    bool        has_model = false;
    std::string model;

    std::istringstream in(text);
    for (std::string line; std::getline(in, line); ) {
        const std::string t = llama_turbot_trim(line);
        if (t.rfind("# verified:", 0) == 0) {
            verified = verified || !llama_turbot_trim(t.substr(std::strlen("# verified:"))).empty();
        } else if (t.rfind("# model:", 0) == 0 && !has_model) {
            has_model = true;
            model     = llama_turbot_trim(t.substr(std::strlen("# model:")));
        }
    }

    if (!verified) {
        why = "no '# verified:' stamp (tools/turbot/turbot_guard.py writes it after the quality gate passes)";
        return false;
    }
    if (!has_model) {
        why = "no '# model:' fingerprint line";
        return false;
    }
    if (fingerprint.empty()) {
        why = "no model to compare the '# model:' fingerprint with";
        return false;
    }
    if (model != llama_turbot_trim(fingerprint)) {
        why = format("'# model: %s' is not this model (%s)", model.c_str(), llama_turbot_trim(fingerprint).c_str());
        return false;
    }

    why.clear();

    return true;
}

// the sidecar text when it passes the check and parses against shape, else false and why
static bool llama_turbot_sidecar_load(const llama_turbot_cache_shape & shape, const std::string & path, std::string & text,
        std::string & why) {
    if (!llama_turbot_plan_read_file(path, text, why)) {
        return false;
    }
    if (!llama_turbot_sidecar_accepts(text, llama_turbot_model_fingerprint(shape), why)) {
        return false;
    }
    llama_turbot_plan plan;
    return llama_turbot_plan_parse_impl(text, path, shape, plan, why, /*quiet =*/ true);
}

//
// [TAG_TURBOT_ANY_PLAN] plan chooser
//

bool llama_turbot_plan_choose(const llama_turbot_cache_shape & shape, llama_turbot_plan_choice & choice, std::string & why) {
    const llama_turbot_switches    sw  = llama_turbot_read_switches();
    const llama_turbot_plan_source src = llama_turbot_plan_get_source();

    llama_turbot_plan plan;
    std::string       text;

    // 1. --kv-tier-plan / LLAMA_TURBOT_PLAN: a file or the built-in plan alone, never an automatic plan on a mismatch
    if (src.origin == LLAMA_TURBOT_PLAN_FILE || src.origin == LLAMA_TURBOT_PLAN_BUILTIN_FORCED) {
        if (!llama_turbot_plan_read(src, text, why)) {
            return false;
        }
        if (!llama_turbot_plan_parse_impl(text, src.name, shape, plan, why, /*quiet =*/ true)) {
            return false;
        }
        choice.kind = src.origin == LLAMA_TURBOT_PLAN_FILE ? LLAMA_TURBOT_PLAN_KIND_FILE : LLAMA_TURBOT_PLAN_KIND_BUILTIN;
        choice.text = text;
        choice.name = src.name;
        why.clear();
        return true;
    }
    if (src.origin == LLAMA_TURBOT_PLAN_AUTO_FORCED) {
        std::string msg;
        if (!llama_turbot_auto_impl(shape, sw, text, msg)) {
            why = format("turbot: plan %s (%s " LLAMA_TURBOT_PLAN_KEYWORD_AUTO "): %s", LLAMA_TURBOT_PLAN_AUTO_NAME,
                    src.set_by == "LLAMA_TURBOT_PLAN" ? "LLAMA_TURBOT_PLAN=" : "--kv-tier-plan", msg.c_str());
            return false;
        }
        if (!llama_turbot_plan_parse_impl(text, LLAMA_TURBOT_PLAN_AUTO_NAME, shape, plan, why, /*quiet =*/ true)) {
            return false;
        }
        choice.kind = LLAMA_TURBOT_PLAN_KIND_AUTO;
        choice.text = text;
        choice.name = LLAMA_TURBOT_PLAN_AUTO_NAME;
        why.clear();
        return true;
    }

    // 2. a verified sidecar of this model
    std::string why_sidecar;
    if (sw.sidecar && shape.model != nullptr) {
        const std::string path = llama_turbot_get_sidecar_path();
        if (!path.empty()) {
            if (llama_turbot_sidecar_load(shape, path, text, why_sidecar)) {
                choice.kind = LLAMA_TURBOT_PLAN_KIND_SIDECAR;
                choice.text = text;
                choice.name = path;
                why.clear();
                return true;
            }
            why_sidecar = format("sidecar %s not used: %s", path.c_str(), why_sidecar.c_str());
        }
    }

    // 3. the built-in plan, when it names exactly these layers with their geometry
    std::string why_builtin;
    text = LLAMA_TURBOT_DEFAULT_PLAN_TEXT;
    if (llama_turbot_plan_parse_impl(text, LLAMA_TURBOT_PLAN_BUILTIN_NAME, shape, plan, why_builtin, /*quiet =*/ true)) {
        choice.kind = LLAMA_TURBOT_PLAN_KIND_BUILTIN;
        choice.text = text;
        choice.name = LLAMA_TURBOT_PLAN_BUILTIN_NAME;
        why.clear();
        return true;
    }

    // 4. the automatic plan
    std::string why_auto;
    if (sw.any) {
        if (!shape.auto_ok) {
            why_auto = "an automatic plan is only made for the main context";
        } else if (!sw.auto_plan) {
            why_auto = "LLAMA_TURBOT_AUTO_PLAN=0";
        } else if (llama_turbot_auto_allowed(shape, sw, why_auto)) {
            std::string msg;
            if (llama_turbot_auto_impl(shape, sw, text, msg)) {
                if (llama_turbot_plan_parse_impl(text, LLAMA_TURBOT_PLAN_AUTO_NAME, shape, plan, why_auto, /*quiet =*/ true)) {
                    choice.kind = LLAMA_TURBOT_PLAN_KIND_AUTO;
                    choice.text = text;
                    choice.name = LLAMA_TURBOT_PLAN_AUTO_NAME;
                    why.clear();
                    return true;
                }
            } else {
                why_auto = msg;
            }
        }
    }

    why = why_builtin;
    if (!why_sidecar.empty()) {
        why += "; " + why_sidecar;
    }
    if (!why_auto.empty()) {
        why += "; no automatic plan: " + why_auto;
    }

    return false;
}

//
// [TAG_TURBOT_ANY_PLAN] plan scope
//

static thread_local const llama_turbot_plan_choice * g_turbot_plan_scope = nullptr;

llama_turbot_plan_scope::llama_turbot_plan_scope(const llama_turbot_plan_choice & c) : choice(c), prev(g_turbot_plan_scope) {
    g_turbot_plan_scope = &choice;
}

llama_turbot_plan_scope::~llama_turbot_plan_scope() {
    g_turbot_plan_scope = prev;
}

const llama_turbot_plan_choice * llama_turbot_plan_scope_current() {
    return g_turbot_plan_scope;
}

//
// the cache constructor's plan loader
//

// the lines of the built-in plan, as before [TAG_TURBOT_ANY_PLAN] (func: "llama_turbot_plan_load")
static void llama_turbot_log_builtin(const char * func, const llama_turbot_plan_source & src) {
    if (src.origin == LLAMA_TURBOT_PLAN_BUILTIN_AUTO) {
        LLAMA_LOG_INFO("%s: turbot: no --kv-tier-plan or LLAMA_TURBOT_PLAN given, using the built-in default plan "
                       "(%s, calibrated on %s)\n", func, LLAMA_TURBOT_DEFAULT_PLAN_FILE, LLAMA_TURBOT_DEFAULT_PLAN_MODEL);
    } else if (src.origin == LLAMA_TURBOT_PLAN_BUILTIN_FORCED) {
        LLAMA_LOG_INFO("%s: turbot: %s: using the built-in default plan (%s, calibrated on %s)\n", func,
                       src.set_by == "LLAMA_TURBOT_PLAN" ? "LLAMA_TURBOT_PLAN=" LLAMA_TURBOT_PLAN_KEYWORD_DEFAULT
                                                         : "--kv-tier-plan " LLAMA_TURBOT_PLAN_KEYWORD_DEFAULT,
                       LLAMA_TURBOT_DEFAULT_PLAN_FILE, LLAMA_TURBOT_DEFAULT_PLAN_MODEL);
    }
}

// [TAG_TURBOT_ANY_PLAN] the INFO line of an automatic plan, and LLAMA_TURBOT_AUTO_PLAN_DUMP
static void llama_turbot_log_auto(const char * func, const llama_turbot_cache_shape & shape, const llama_turbot_plan & plan,
        const std::string & text, const llama_turbot_plan_source & src) {
    const llama_turbot_switches sw = llama_turbot_read_switches();

    std::map<int32_t, llama_turbot_shape_geom> geoms;
    std::string msg;
    llama_turbot_shape_geoms(shape, sw.any, geoms, msg);

    const uint64_t            kv_cells = llama_turbot_shape_cells(shape);
    const llama_turbot_budget budget   = llama_turbot_budget_of(geoms, kv_cells, sw.auto_budget_turbo5p);

    const std::string head = src.origin == LLAMA_TURBOT_PLAN_AUTO_FORCED
        ? format("%s: automatic uncalibrated plan", src.set_by == "LLAMA_TURBOT_PLAN" ? "LLAMA_TURBOT_PLAN=" LLAMA_TURBOT_PLAN_KEYWORD_AUTO
                                                                                     : "--kv-tier-plan " LLAMA_TURBOT_PLAN_KEYWORD_AUTO)
        : format("no plan names this model's %d attention layers; automatic uncalibrated plan", (int) plan.layers.size());

    LLAMA_LOG_INFO("%s: turbot: %s: old widths 4/5 (mean %.3f), young %d, POOL %u, CAP %u, %.2f MiB (%s %.2f MiB), hash 0x%016llx\n",
            func, head.c_str(), llama_turbot_plan_mean_width(plan), GGML_TURBOT_Y_DEFAULT, plan.pool_cells, plan.cap_cells,
            llama_turbot_plan_bytes(plan, kv_cells)/LLAMA_TURBOT_MIB, budget.name.c_str(), budget.bytes/LLAMA_TURBOT_MIB,
            (unsigned long long) plan.hash);

    const char * dump = getenv("LLAMA_TURBOT_AUTO_PLAN_DUMP");
    if (dump != nullptr && dump[0] != '\0') {
        std::ofstream f(dump, std::ios::binary);
        f << text;
        if (!f) {
            LLAMA_LOG_WARN("%s: turbot: LLAMA_TURBOT_AUTO_PLAN_DUMP: cannot write %s\n", func, dump);
        } else {
            LLAMA_LOG_INFO("%s: turbot: automatic plan written to %s (LLAMA_TURBOT_AUTO_PLAN_DUMP)\n", func, dump);
        }
    }
}

// [TAG_TURBOT_ANY_SIDECAR] a sidecar that was set but not used says why, once per path
static void llama_turbot_log_sidecar_unused(const char * func, const llama_turbot_cache_shape & shape, llama_turbot_plan_kind used,
        bool scoped) {
    if (used == LLAMA_TURBOT_PLAN_KIND_SIDECAR || !llama_turbot_read_switches().sidecar) {
        return;
    }
    const std::string path = llama_turbot_get_sidecar_path();
    if (path.empty()) {
        return;
    }
    {
        static std::mutex            mutex;
        static std::set<std::string> reported;
        std::lock_guard<std::mutex>  lock(mutex);
        if (!reported.insert(path).second) {
            return;
        }
    }

    std::string why;
    const llama_turbot_plan_source src = llama_turbot_plan_get_source();
    if (src.origin == LLAMA_TURBOT_PLAN_FILE || src.origin == LLAMA_TURBOT_PLAN_BUILTIN_FORCED || src.origin == LLAMA_TURBOT_PLAN_AUTO_FORCED) {
        why = format("%s is given", src.set_by.c_str());
    } else if (!scoped) {
        why = "without a plan chosen by llama_context (LLAMA_KV_RESOLVE=0) only --kv-tier-plan, LLAMA_TURBOT_PLAN and the built-in plan are used";
    } else if (shape.model == nullptr) {
        why = "no model to compare its '# model:' fingerprint with";
    } else {
        std::string text;
        if (llama_turbot_sidecar_load(shape, path, text, why)) {
            why = "the plan was chosen before this cache was built";
        }
    }

    LLAMA_LOG_INFO("%s: turbot: sidecar plan %s not used: %s\n", func, path.c_str(), why.c_str());
}

// [TAG_TURBOT_ANY_SIDECAR] what a verified plan for this model carries as '# model:' (tools/turbot/turbot_guard.py).
// LLAMA_TURBOT_ANY=0: no line (a plan file then logs exactly what it logged before [TAG_TURBOT_ANY_*]).
static void llama_turbot_log_fingerprint(const char * func, const llama_turbot_cache_shape & shape) {
    if (!llama_turbot_read_switches().any) {
        return;
    }
    const std::string fp = llama_turbot_model_fingerprint(shape);
    if (!fp.empty()) {
        LLAMA_LOG_INFO("%s: turbot: model fingerprint: %s\n", func, fp.c_str());
    }
}

bool llama_turbot_plan_load(const llama_turbot_cache_shape & shape, llama_turbot_plan & plan, std::string & err) {
    llama_turbot_plan res;

    // [TAG_TURBOT_ANY_PLAN] the plan llama_context chose for this context
    if (const llama_turbot_plan_choice * scope = llama_turbot_plan_scope_current()) {
        if (!llama_turbot_plan_parse_impl(scope->text, scope->name, shape, res, err)) {
            return false;
        }
        res.path = scope->name;

        const llama_turbot_plan_source src = llama_turbot_plan_get_source();
        switch (scope->kind) {
            case LLAMA_TURBOT_PLAN_KIND_BUILTIN:
                llama_turbot_log_builtin(__func__, src);
                break;
            case LLAMA_TURBOT_PLAN_KIND_SIDECAR:
                LLAMA_LOG_INFO("%s: turbot: using the verified sidecar plan %s\n", __func__, scope->name.c_str());
                llama_turbot_log_fingerprint(__func__, shape);
                break;
            case LLAMA_TURBOT_PLAN_KIND_AUTO:
                llama_turbot_log_auto(__func__, shape, res, scope->text, src);
                llama_turbot_log_fingerprint(__func__, shape);
                break;
            case LLAMA_TURBOT_PLAN_KIND_FILE:
                llama_turbot_log_fingerprint(__func__, shape);
                break;
        }
        llama_turbot_log_sidecar_unused(__func__, shape, scope->kind, true);

        plan = std::move(res);

        return true;
    }

    const llama_turbot_plan_source src = llama_turbot_plan_get_source();

    // [TAG_TURBOT_ANY_PLAN] "auto" without a scope (LLAMA_KV_RESOLVE=0): generated from this cache's shape
    if (src.origin == LLAMA_TURBOT_PLAN_AUTO_FORCED) {
        std::string text;
        std::string msg;
        if (!llama_turbot_auto_impl(shape, llama_turbot_read_switches(), text, msg)) {
            err = format("turbot: plan %s (%s " LLAMA_TURBOT_PLAN_KEYWORD_AUTO "): %s", LLAMA_TURBOT_PLAN_AUTO_NAME,
                    src.set_by == "LLAMA_TURBOT_PLAN" ? "LLAMA_TURBOT_PLAN=" : "--kv-tier-plan", msg.c_str());
            return false;
        }
        if (!llama_turbot_plan_parse_impl(text, LLAMA_TURBOT_PLAN_AUTO_NAME, shape, res, err)) {
            return false;
        }
        res.path = LLAMA_TURBOT_PLAN_AUTO_NAME;

        llama_turbot_log_auto(__func__, shape, res, text, src);
        llama_turbot_log_fingerprint(__func__, shape);
        llama_turbot_log_sidecar_unused(__func__, shape, LLAMA_TURBOT_PLAN_KIND_AUTO, false);

        plan = std::move(res);

        return true;
    }

    std::string text;
    if (!llama_turbot_plan_read(src, text, err)) {
        return false;
    }

    if (!llama_turbot_plan_parse_impl(text, src.name, shape, res, err)) {
        if (src.origin != LLAMA_TURBOT_PLAN_FILE) {
            err += format(" (the built-in plan was calibrated on %s and covers attention layers %s; for this model pass "
                          "--kv-tier-plan <file> with a plan for its layers%s, or use -ctk turbo5p -ctv turbo5p)",
                          LLAMA_TURBOT_DEFAULT_PLAN_MODEL, LLAMA_TURBOT_DEFAULT_PLAN_LAYERS,
                          llama_turbot_read_switches().any ? " or --kv-tier-plan " LLAMA_TURBOT_PLAN_KEYWORD_AUTO : "");
        }
        return false;
    }
    res.path = src.name;

    llama_turbot_log_builtin(__func__, src);
    if (src.origin == LLAMA_TURBOT_PLAN_FILE) {
        llama_turbot_log_fingerprint(__func__, shape);
    }
    llama_turbot_log_sidecar_unused(__func__, shape,
            src.origin == LLAMA_TURBOT_PLAN_FILE ? LLAMA_TURBOT_PLAN_KIND_FILE : LLAMA_TURBOT_PLAN_KIND_BUILTIN, false);

    plan = std::move(res);

    return true;
}

bool llama_turbot_plan_load(const std::vector<int32_t> & attn_layers, uint32_t kv_size, llama_turbot_plan & plan, std::string & err) {
    return llama_turbot_plan_load(llama_turbot_shape_4x256(attn_layers, kv_size), plan, err);
}

//
// [TAG_TURBOT] llama_kv_tier (SPEC 9.5-9.8)
//

llama_kv_tier::llama_kv_tier(uint32_t kv_size, uint32_t pool_cells, uint32_t cap_cells) :
    kv_size(kv_size), pool_cells(pool_cells), cap_cells(cap_cells),
    n_gran(kv_size / GGML_TURBOT_GRANULE), n_slot(pool_cells / GGML_TURBOT_GRANULE) {
    GGML_ASSERT(kv_size % GGML_TURBOT_GRANULE == 0);
    GGML_ASSERT(pool_cells % GGML_TURBOT_GRANULE == 0 && pool_cells <= kv_size);

    gslot        .assign(n_gran, -1);
    owner        .assign(n_slot, -1);
    last_touch   .assign(n_slot, 0);
    ref_valid    .assign(n_gran, 0);
    stamps       .assign(kv_size, 0);
    row_ctr      .assign(LLAMA_MAX_SEQ, 0);
    cut          .assign(LLAMA_MAX_SEQ, 0);
    in_flight_pos.assign(n_gran, 0);

    for (int32_t s = 0; s < (int32_t) n_slot; ++s) {
        free_slots.insert(free_slots.end(), s);
    }

    const char * LLAMA_TURBOT_DEBUG = getenv("LLAMA_TURBOT_DEBUG");
    debug = LLAMA_TURBOT_DEBUG ? atoi(LLAMA_TURBOT_DEBUG) : 0;
}

uint32_t llama_kv_tier::n_granules() const {
    return n_gran;
}

uint32_t llama_kv_tier::n_slots() const {
    return n_slot;
}

const std::vector<int32_t> & llama_kv_tier::granule_slots() const {
    return gslot;
}

bool llama_kv_tier::cell_young(uint32_t cell) const {
    GGML_ASSERT(cell < kv_size);

    const uint32_t g = cell >> GGML_TURBOT_LOG2_GRANULE;
    const uint64_t b = 1ull << (cell & (GGML_TURBOT_GRANULE - 1));

    return gslot[g] >= 0 && (ref_valid[g] & b) != 0 && (pending_of(g) & b) == 0;
}

const std::vector<int32_t> & llama_kv_tier::young_rows() const {
    return young;
}

const std::vector<int32_t> & llama_kv_tier::fill_entries() const {
    return fill;
}

uint64_t llama_kv_tier::live_mask(const llama_kv_cells & kvc, uint32_t g) const {
    const uint32_t c0 = g*GGML_TURBOT_GRANULE;

    uint64_t m = 0;
    for (uint32_t c = 0; c < GGML_TURBOT_GRANULE; ++c) {
        if (!kvc.is_empty(c0 + c)) {
            m |= 1ull << c;
        }
    }

    return m;
}

void llama_kv_tier::pending_set(uint32_t g, uint64_t mask) {
    if (mask != 0) {
        pending[g] |= mask;
    }
}

void llama_kv_tier::pending_clear(uint32_t g, uint64_t mask) {
    const auto it = pending.find(g);
    if (it != pending.end()) {
        it->second &= ~mask;
        if (it->second == 0) {
            pending.erase(it);
        }
    }
}

uint64_t llama_kv_tier::pending_of(uint32_t g) const {
    const auto it = pending.find(g);
    return it == pending.end() ? 0 : it->second;
}

void llama_kv_tier::clear_in_flight() {
    for (const uint32_t g : in_flight) {
        in_flight_pos[g] = 0;
    }
    in_flight.clear();
}

// aging is metadata only: the granule reads through its base code from the next graph on, the pool rows are reused
void llama_kv_tier::free_slot(int32_t slot, bool to_free_set) {
    const int32_t g = owner[slot];
    if (g >= 0) {
        gslot[g]     = -1;
        ref_valid[g] = 0;
        pending.erase((uint32_t) g);
    }
    owner[slot] = -1;

    if (to_free_set) {
        free_slots.insert(slot);
    }
}

// eviction order (SPEC 9.6): empty granules first, then the smallest stamp margin over the owners' cuts, then the
// oldest touch, then the lowest slot. Granules in flight, and during a restore the slots it allocated, are protected.
void llama_kv_tier::build_victims(const llama_kv_cells & kvc, bool during_restore) {
    victims.clear();
    victims_next  = 0;
    victims_built = true;

    for (int32_t slot = 0; slot < (int32_t) n_slot; ++slot) {
        const int32_t g = owner[slot];
        if (g < 0 || in_flight_pos[g] != 0) {
            continue;
        }
        if (during_restore && std::find(restored.begin(), restored.end(), slot) != restored.end()) {
            continue;
        }

        victim v;
        v.cls    = 0;
        v.margin = 0;
        v.touch  = last_touch[slot];
        v.slot   = slot;

        int64_t margin = std::numeric_limits<int64_t>::min();

        const uint32_t c0 = (uint32_t) g*GGML_TURBOT_GRANULE;
        for (uint32_t c = 0; c < GGML_TURBOT_GRANULE; ++c) {
            const uint32_t cell = c0 + c;
            if (kvc.is_empty(cell)) {
                continue;
            }
            v.cls = 1;

            const int64_t st = (int64_t) stamps[cell];
            llama_turbot_for_each_seq(kvc.seq_bits(cell), [&](llama_seq_id s) {
                if (margin == std::numeric_limits<int64_t>::max()) {
                    return;
                }
                if (!has_cut.test(s)) {
                    // the sequence appeared after the last commit: its whole band is wanted
                    margin = std::numeric_limits<int64_t>::max();
                    return;
                }
                margin = std::max(margin, st - cut[s]);
            });

            if (margin == std::numeric_limits<int64_t>::max()) {
                break;
            }
        }

        if (v.cls == 1) {
            v.margin = margin;
        }

        victims.push_back(v);
    }

    std::sort(victims.begin(), victims.end(), [](const victim & a, const victim & b) {
        if (a.cls    != b.cls)    { return a.cls    < b.cls;    }
        if (a.margin != b.margin) { return a.margin < b.margin; }
        if (a.touch  != b.touch)  { return a.touch  < b.touch;  }
        return a.slot < b.slot;
    });
}

int32_t llama_kv_tier::alloc_slot(const llama_kv_cells & kvc, bool during_restore) {
    if (!free_slots.empty()) {
        const int32_t slot = *free_slots.begin();
        free_slots.erase(free_slots.begin());
        return slot;
    }

    if (!victims_built) {
        build_victims(kvc, during_restore);
    }

    while (victims_next < victims.size()) {
        const victim & v = victims[victims_next++];

        const int32_t g = owner[v.slot];
        if (g < 0 || in_flight_pos[g] != 0) {
            continue;
        }
        if (during_restore && std::find(restored.begin(), restored.end(), v.slot) != restored.end()) {
            continue;
        }

        if (v.cls == 0) {
            dbg_reclaimed++;
        } else {
            n_evict++;
            dbg_evicted++;
        }

        free_slot(v.slot, false);

        return v.slot;
    }

    return -1;
}

void llama_kv_tier::begin_ubatch(const llama_ubatch & ubatch, const std::vector<uint32_t> & cells, const llama_kv_cells & kvc) {
    GGML_ASSERT(cells.size() == ubatch.n_tokens);

    if (!in_flight.empty()) {
        // defensive: the previous ubatch was neither committed nor aborted
        abort_ubatch();
    }

    restored.clear();
    restored_cells.clear();

    ++touch_serial;

    victims.clear();
    victims_built = false;
    victims_next  = 0;

    fill.clear();

    const uint32_t n_rows = ubatch.n_tokens;

    young.assign(n_rows, -1);

    // per-sequence write-row stamps: a row listed in several sequences advances every listed counter and takes its
    // stamp from the first
    for (uint32_t i = 0; i < n_rows; ++i) {
        GGML_ASSERT(cells[i] < kv_size);
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            GGML_ASSERT(s >= 0 && s < (llama_seq_id) LLAMA_MAX_SEQ);
            row_ctr[s]++;
        }
        stamps[cells[i]] = row_ctr[ubatch.seq_id[i][0]];
    }

    // granules touched by this ubatch, in first-touch order, with the cells it writes into each
    std::vector<uint64_t> written;
    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t g = cells[i] >> GGML_TURBOT_LOG2_GRANULE;
        if (in_flight_pos[g] == 0) {
            in_flight.push_back(g);
            written.push_back(0);
            in_flight_pos[g] = (uint32_t) in_flight.size();
        }
        written[in_flight_pos[g] - 1] |= 1ull << (cells[i] & (GGML_TURBOT_GRANULE - 1));
    }

    const auto visit = [&](uint32_t g, bool touched, uint64_t wr) {
        if (gslot[g] < 0 && touched) {
            const int32_t slot = alloc_slot(kvc, false);
            if (slot < 0) {
                // every slot belongs to a granule of this ubatch: the rows of g are written old-only
                if (n_slot > 0 && (dbg_no_slot++ % 1000) == 0) {
                    LLAMA_LOG_WARN("%s: turbot: every young slot is in flight, granule %u is written old-only (%" PRIu64 " times so far)\n",
                            __func__, g, dbg_no_slot);
                }
                return;
            }
            gslot[g]     = slot;
            owner[slot]  = (int32_t) g;
            ref_valid[g] = 0;
        }

        if (gslot[g] < 0) {
            pending.erase(g);
            return;
        }

        if (touched) {
            last_touch[gslot[g]] = touch_serial;
        }

        const uint64_t live = live_mask(kvc, g);

        // stale bits of emptied cells never survive (9.8)
        ref_valid[g] &= live;

        // live cells without a refinement, other than the ones this ubatch writes, get a center fill
        const uint64_t need = live & ~ref_valid[g] & ~wr;
        if (need != 0) {
            fill.push_back((int32_t) g);
            fill.push_back(gslot[g]);
            fill.push_back((int32_t) (uint32_t) (need & 0xffffffffull));
            fill.push_back((int32_t) (uint32_t) (need >> 32));
            ref_valid[g] |= need;
            if (debug > 0) {
                dbg_fill_cells += (uint64_t) llama_turbot_popcount(need);
            }
        }

        pending.erase(g);
    };

    for (size_t k = 0; k < in_flight.size(); ++k) {
        visit(in_flight[k], true, written[k]);
    }

    if (!pending.empty()) {
        std::vector<uint32_t> keys;
        for (const auto & it : pending) {
            if (in_flight_pos[it.first] == 0) {
                keys.push_back(it.first);
            }
        }
        for (const uint32_t g : keys) {
            visit(g, false, 0);
        }
    }

    for (uint32_t i = 0; i < n_rows; ++i) {
        const uint32_t g = cells[i] >> GGML_TURBOT_LOG2_GRANULE;
        if (gslot[g] >= 0) {
            young[i] = ggml_turbot_pool_row(gslot[g], cells[i]);
            ref_valid[g] |= 1ull << (cells[i] & (GGML_TURBOT_GRANULE - 1));
        }
    }

    victims.clear();
    victims_built = false;

    if (debug >= 2) {
        check_invariant(kvc, true, __func__);
    }
}

void llama_kv_tier::commit_ubatch(const llama_kv_cells & kvc) {
    // quota (SPEC 9.6)
    uint32_t                   n_cells[LLAMA_MAX_SEQ];
    std::bitset<LLAMA_MAX_SEQ> live;

    uint32_t n_active = 0;
    uint64_t sum      = 0;

    for (llama_seq_id s = 0; s < (llama_seq_id) LLAMA_MAX_SEQ; ++s) {
        n_cells[s] = kvc.seq_n_cells(s);
        if (n_cells[s] > 0) {
            live.set(s);
            n_active++;
            sum += n_cells[s];
        }
    }

    const int64_t n_eff = std::max<int64_t>(0,
            (int64_t) pool_cells - (int64_t) GGML_TURBOT_GRANULE*GGML_TURBOT_QUOTA_SLACK_GRANULES*(int64_t) n_active);

    llama_turbot_for_each_seq(live, [&](llama_seq_id s) {
        const uint64_t y = sum ? std::min<uint64_t>(cap_cells, (uint64_t) n_eff*n_cells[s]/sum) : 0;
        cut[s] = (int64_t) row_ctr[s] - (int64_t) y;
    });
    has_cut = live;

    uint64_t n_freed = 0;

    for (int32_t slot = 0; slot < (int32_t) n_slot; ++slot) {
        const int32_t g = owner[slot];
        if (g < 0) {
            continue;
        }

        bool wanted   = false;
        bool any_live = false;

        const uint32_t c0 = (uint32_t) g*GGML_TURBOT_GRANULE;
        for (uint32_t c = 0; c < GGML_TURBOT_GRANULE && !wanted; ++c) {
            const uint32_t cell = c0 + c;
            if (kvc.is_empty(cell)) {
                continue;
            }
            any_live = true;

            const int64_t st = (int64_t) stamps[cell];
            llama_turbot_for_each_seq(kvc.seq_bits(cell) & live, [&](llama_seq_id s) {
                if (st > cut[s]) {
                    wanted = true;
                }
            });
        }

        if (!any_live || (!wanted && in_flight_pos[g] == 0)) {
            free_slot(slot, true);
            n_freed++;
        }
    }

    if (debug >= 1) {
        uint32_t young_n[LLAMA_MAX_SEQ] = {};
        for (int32_t slot = 0; slot < (int32_t) n_slot; ++slot) {
            const int32_t g = owner[slot];
            if (g < 0) {
                continue;
            }
            for (uint32_t c = 0; c < GGML_TURBOT_GRANULE; ++c) {
                const uint32_t cell = (uint32_t) g*GGML_TURBOT_GRANULE + c;
                if (!kvc.is_empty(cell) && cell_young(cell)) {
                    llama_turbot_for_each_seq(kvc.seq_bits(cell), [&](llama_seq_id s) { young_n[s]++; });
                }
            }
        }

        std::string per_seq;
        llama_turbot_for_each_seq(live, [&](llama_seq_id s) {
            per_seq += format(" s%d %u/%u", s, young_n[s], n_cells[s]);
        });

        LLAMA_LOG_INFO("%s: turbot: slots used %u/%u, freed %" PRIu64 ", empty reclaimed %" PRIu64 ", evicted %" PRIu64
                ", filled cells %" PRIu64 ", young/live:%s\n", __func__,
                n_slot - (uint32_t) free_slots.size(), n_slot, n_freed, dbg_reclaimed, dbg_evicted, dbg_fill_cells, per_seq.c_str());

        dbg_reclaimed  = 0;
        dbg_evicted    = 0;
        dbg_fill_cells = 0;
    }

    clear_in_flight();

    restored.clear();
    restored_cells.clear();
}

void llama_kv_tier::abort_ubatch() {
    if (in_flight.empty()) {
        // nothing was begun since the last commit or abort
        return;
    }

    // the fills of this ubatch may not have run: take them back and queue them again. No slot is freed; the next
    // commit reclaims what is no longer wanted.
    for (size_t k = 0; k + 3 < fill.size(); k += 4) {
        const uint32_t g    = (uint32_t) fill[k];
        const uint64_t mask = (uint64_t) (uint32_t) fill[k + 2] | ((uint64_t) (uint32_t) fill[k + 3] << 32);

        ref_valid[g] &= ~mask;
        if (gslot[g] >= 0) {
            pending_set(g, mask);
        }
    }

    clear_in_flight();
}

void llama_kv_tier::on_cell_emptied(uint32_t cell) {
    GGML_ASSERT(cell < kv_size);

    const uint32_t g = cell >> GGML_TURBOT_LOG2_GRANULE;
    const uint64_t b = 1ull << (cell & (GGML_TURBOT_GRANULE - 1));

    ref_valid[g] &= ~b;
    pending_clear(g, b);
}

void llama_kv_tier::on_seq_tail_removed(llama_seq_id seq, uint64_t min_removed_stamp) {
    GGML_ASSERT(seq >= 0 && seq < (llama_seq_id) LLAMA_MAX_SEQ);

    row_ctr[seq] = min_removed_stamp > 0 ? std::min(row_ctr[seq], min_removed_stamp - 1) : 0;
}

void llama_kv_tier::on_seq_cp(llama_seq_id src, llama_seq_id dst, bool dst_was_empty) {
    GGML_ASSERT(src >= 0 && src < (llama_seq_id) LLAMA_MAX_SEQ);
    GGML_ASSERT(dst >= 0 && dst < (llama_seq_id) LLAMA_MAX_SEQ);

    row_ctr[dst] = dst_was_empty ? row_ctr[src] : std::max(row_ctr[dst], row_ctr[src]);
}

void llama_kv_tier::clear() {
    gslot     .assign(n_gran, -1);
    owner     .assign(n_slot, -1);
    last_touch.assign(n_slot, 0);
    ref_valid .assign(n_gran, 0);
    stamps    .assign(kv_size, 0);
    row_ctr   .assign(LLAMA_MAX_SEQ, 0);
    cut       .assign(LLAMA_MAX_SEQ, 0);
    has_cut.reset();

    free_slots.clear();
    for (int32_t s = 0; s < (int32_t) n_slot; ++s) {
        free_slots.insert(free_slots.end(), s);
    }

    touch_serial = 0;

    clear_in_flight();

    restored.clear();
    restored_cells.clear();
    pending.clear();

    victims.clear();
    victims_built = false;
    victims_next  = 0;

    young.clear();
    fill.clear();
}

std::vector<int32_t> llama_kv_tier::restore_cells(const std::vector<uint32_t> & cells, const std::vector<uint64_t> & stamps_in,
                                                  const std::vector<uint8_t> & young_in, const std::vector<std::pair<llama_seq_id, uint64_t>> & counters,
                                                  const llama_kv_cells & kvc) {
    GGML_ASSERT(stamps_in.size() == cells.size());
    GGML_ASSERT(young_in.size()  == cells.size());

    restored.clear();
    restored_cells = cells;

    ++touch_serial;

    victims.clear();
    victims_built = false;
    victims_next  = 0;

    // an assignment, never max(): the destination may be any slot id, whose own counter says nothing about the blob
    for (const auto & [s, v] : counters) {
        GGML_ASSERT(s >= 0 && s < (llama_seq_id) LLAMA_MAX_SEQ);
        row_ctr[s] = v;
    }

    for (size_t i = 0; i < cells.size(); ++i) {
        GGML_ASSERT(cells[i] < kv_size);
        stamps[cells[i]] = stamps_in[i];
    }

    // granules receiving at least one young cell, ascending
    std::vector<uint32_t> gs;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (young_in[i]) {
            gs.push_back(cells[i] >> GGML_TURBOT_LOG2_GRANULE);
        }
    }
    std::sort(gs.begin(), gs.end());
    gs.erase(std::unique(gs.begin(), gs.end()), gs.end());

    std::vector<uint32_t> newly;   // ascending, like gs

    for (const uint32_t g : gs) {
        if (gslot[g] < 0) {
            const int32_t slot = alloc_slot(kvc, true);
            if (slot < 0) {
                continue;
            }
            gslot[g]     = slot;
            owner[slot]  = (int32_t) g;
            ref_valid[g] = 0;
            pending.erase(g);

            restored.push_back(slot);
            newly.push_back(g);
        }
        last_touch[gslot[g]] = touch_serial;
    }

    std::vector<int32_t> rows(cells.size(), -1);

    std::map<uint32_t, uint64_t> restored_mask;   // newly allocated granule -> its restored cells

    for (size_t i = 0; i < cells.size(); ++i) {
        const uint32_t c = cells[i];
        const uint32_t g = c >> GGML_TURBOT_LOG2_GRANULE;
        const uint64_t b = 1ull << (c & (GGML_TURBOT_GRANULE - 1));

        if (young_in[i] && gslot[g] >= 0) {
            rows[i] = ggml_turbot_pool_row(gslot[g], c);
            ref_valid[g] |= b;
            pending_clear(g, b);
        } else {
            ref_valid[g] &= ~b;
            if (gslot[g] >= 0) {
                pending_set(g, b);
            }
        }

        if (std::binary_search(newly.begin(), newly.end(), g)) {
            restored_mask[g] |= b;
        }
    }

    // a granule this call made young may already hold live cells that have only old codes
    for (const uint32_t g : newly) {
        if (gslot[g] < 0) {
            continue;
        }
        const uint64_t others = live_mask(kvc, g) & ~restored_mask[g];
        ref_valid[g] &= ~others;
        pending_set(g, others);
    }

    victims.clear();
    victims_built = false;

    if (debug >= 2) {
        check_invariant(kvc, false, __func__);
    }

    return rows;
}

void llama_kv_tier::abort_restore() {
    for (const int32_t slot : restored) {
        if (owner[slot] >= 0) {
            free_slot(slot, true);
        }
    }

    // granules that were young before the call: the restored cells have no refinement bytes. The pending bit keeps
    // 9.8 (1) until the failure path removes the cells, which clears both bits.
    for (const uint32_t c : restored_cells) {
        const uint32_t g = c >> GGML_TURBOT_LOG2_GRANULE;
        if (gslot[g] < 0) {
            continue;
        }
        const uint64_t b = 1ull << (c & (GGML_TURBOT_GRANULE - 1));
        ref_valid[g] &= ~b;
        pending_set(g, b);
    }

    restored.clear();
    restored_cells.clear();
}

uint64_t llama_kv_tier::stamp(uint32_t cell) const {
    GGML_ASSERT(cell < kv_size);
    return stamps[cell];
}

uint64_t llama_kv_tier::row_counter(llama_seq_id seq) const {
    GGML_ASSERT(seq >= 0 && seq < (llama_seq_id) LLAMA_MAX_SEQ);
    return row_ctr[seq];
}

uint64_t llama_kv_tier::n_evictions() const {
    return n_evict;
}

bool llama_kv_tier::seq_cut(llama_seq_id seq, int64_t & cut_out) const {
    GGML_ASSERT(seq >= 0 && seq < (llama_seq_id) LLAMA_MAX_SEQ);
    cut_out = cut[seq];
    return has_cut.test(seq);
}

// [TAG_TURBOT_ANY_STREAMS]
void llama_kv_tier::copy_stream_from(const llama_kv_tier & src, llama_seq_id seq_src, llama_seq_id seq_dst) {
    GGML_ASSERT(&src != this);
    GGML_ASSERT(src.kv_size == kv_size && src.pool_cells == pool_cells);
    GGML_ASSERT(seq_src >= 0 && seq_src < (llama_seq_id) LLAMA_MAX_SEQ);
    GGML_ASSERT(seq_dst >= 0 && seq_dst < (llama_seq_id) LLAMA_MAX_SEQ);

    // the granule table and every per-cell bit: the pool-slice copy gives this stream's pool rows the source's bytes
    gslot        = src.gslot;
    owner        = src.owner;
    free_slots   = src.free_slots;
    last_touch   = src.last_touch;
    touch_serial = src.touch_serial;
    ref_valid    = src.ref_valid;
    stamps       = src.stamps;
    pending      = src.pending;

    // the destination stream was reset: only seq_dst has cells, with the source sequence's counter and cut
    const uint64_t ctr     = src.row_ctr[seq_src];
    const int64_t  cut_src = src.cut[seq_src];
    const bool     has_src = src.has_cut.test(seq_src);

    row_ctr.assign(LLAMA_MAX_SEQ, 0);
    cut    .assign(LLAMA_MAX_SEQ, 0);
    has_cut.reset();

    row_ctr[seq_dst] = ctr;
    cut    [seq_dst] = cut_src;
    has_cut.set(seq_dst, has_src);

    clear_in_flight();

    restored.clear();
    restored_cells.clear();

    victims.clear();
    victims_built = false;
    victims_next  = 0;

    young.clear();
    fill.clear();
}

void llama_kv_tier::drop_empty_cells(const llama_kv_cells & kvc) {
    for (uint32_t g = 0; g < n_gran; ++g) {
        if (gslot[g] < 0) {
            continue;
        }
        const uint64_t live = live_mask(kvc, g);
        ref_valid[g] &= live;
        pending_clear(g, ~live);
    }
}

//
// [TAG_TURBOT_ANY_STREAMS] the tiers of a multi-stream cache (llama-kv-tier.h)
//

void llama_turbot_streams_begin(llama_kv_tier_ptrs & tiers, const llama_ubatch & ubatch, const std::vector<llama_seq_id> & strm,
        const std::vector<std::vector<uint32_t>> & idxs, const std::vector<llama_kv_cells> & cells) {
    GGML_ASSERT(strm.size() == idxs.size() && !strm.empty());

    // one stream: the whole ubatch, exactly the call of a single-stream cache
    if (strm.size() == 1) {
        GGML_ASSERT(strm[0] >= 0 && (size_t) strm[0] < tiers.size());
        tiers[strm[0]]->begin_ubatch(ubatch, idxs[0], cells[strm[0]]);
        return;
    }

    // split_equal: rows [k*n, (k+1)*n) belong to stream strm[k]
    const uint32_t n = (uint32_t) idxs[0].size();
    GGML_ASSERT(ubatch.n_tokens == n*strm.size());

    for (size_t k = 0; k < strm.size(); ++k) {
        GGML_ASSERT(strm[k] >= 0 && (size_t) strm[k] < tiers.size() && idxs[k].size() == n);

        // begin_ubatch reads n_tokens, n_seq_id and seq_id only; the other fields keep describing the whole ubatch
        llama_ubatch slice = ubatch;
        slice.n_tokens     = n;
        slice.n_seq_tokens = n;
        slice.n_seqs       = 1;
        slice.n_seq_id     = ubatch.n_seq_id + k*n;
        slice.seq_id       = ubatch.seq_id   + k*n;

        tiers[strm[k]]->begin_ubatch(slice, idxs[k], cells[strm[k]]);
    }
}

void llama_turbot_streams_commit(llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, const std::vector<llama_kv_cells> & cells) {
    for (const llama_seq_id s : strm) {
        GGML_ASSERT(s >= 0 && (size_t) s < tiers.size());
        tiers[s]->commit_ubatch(cells[s]);
    }
}

void llama_turbot_streams_abort(llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm) {
    for (const llama_seq_id s : strm) {
        GGML_ASSERT(s >= 0 && (size_t) s < tiers.size());
        tiers[s]->abort_ubatch();
    }
}

int64_t llama_turbot_streams_n_fill(const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm) {
    int64_t n = 0;
    for (const llama_seq_id s : strm) {
        GGML_ASSERT(s >= 0 && (size_t) s < tiers.size());
        n += (int64_t) tiers[s]->fill_entries().size()/4;
    }
    return n;
}

void llama_turbot_streams_gtab(const llama_kv_tier_ptrs & tiers, uint32_t s0, uint32_t s1, int32_t * dst) {
    GGML_ASSERT(s0 <= s1 && s1 < tiers.size());

    const uint32_t n_gran   = tiers[0]->n_granules();
    const int32_t  n_slot_s = (int32_t) tiers[0]->n_slots();

    for (uint32_t s = s0; s <= s1; ++s) {
        const auto & gs  = tiers[s]->granule_slots();
        const int32_t off = (int32_t) s*n_slot_s;
        int32_t * out = dst + (size_t) (s - s0)*n_gran;
        for (uint32_t g = 0; g < n_gran; ++g) {
            out[g] = gs[g] >= 0 ? gs[g] + off : -1;
        }
    }
}

void llama_turbot_streams_young(const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, int32_t * dst) {
    const int32_t pool_s = (int32_t) tiers[0]->n_slots()*GGML_TURBOT_GRANULE;

    size_t i = 0;
    for (const llama_seq_id s : strm) {
        GGML_ASSERT(s >= 0 && (size_t) s < tiers.size());
        const int32_t off = (int32_t) s*pool_s;
        for (const int32_t row : tiers[s]->young_rows()) {
            dst[i++] = row >= 0 ? row + off : -1;
        }
    }
}

void llama_turbot_streams_fill(const llama_kv_tier_ptrs & tiers, const std::vector<llama_seq_id> & strm, int32_t * dst) {
    const int32_t n_gran   = (int32_t) tiers[0]->n_granules();
    const int32_t n_slot_s = (int32_t) tiers[0]->n_slots();

    size_t i = 0;
    for (const llama_seq_id s : strm) {
        GGML_ASSERT(s >= 0 && (size_t) s < tiers.size());
        const auto & f = tiers[s]->fill_entries();
        for (size_t k = 0; k + 3 < f.size(); k += 4) {
            dst[i++] = f[k]     + s*n_gran;
            dst[i++] = f[k + 1] + s*n_slot_s;
            dst[i++] = f[k + 2];
            dst[i++] = f[k + 3];
        }
    }
}

// SPEC 9.8, env LLAMA_TURBOT_DEBUG=2
void llama_kv_tier::check_invariant(const llama_kv_cells & kvc, bool after_begin, const char * where) const {
    uint32_t n_owned = 0;

    for (uint32_t g = 0; g < n_gran; ++g) {
        const uint64_t pend = pending_of(g);

        if (gslot[g] < 0) {
            if (ref_valid[g] != 0 || pend != 0) {
                LLAMA_LOG_ERROR("%s: turbot invariant violated: old granule %u has ref_valid %016" PRIx64 " pending %016" PRIx64 "\n",
                        where, g, ref_valid[g], pend);
                GGML_ABORT("turbot: tier invariant violated");
            }
            continue;
        }

        n_owned++;

        const uint64_t live = live_mask(kvc, g);

        const bool ok_owner = owner[gslot[g]] == (int32_t) g;
        const bool ok_1     = ((live & ~pend) & ~ref_valid[g]) == 0;
        const bool ok_2     = (~live & (ref_valid[g] | pend)) == 0;
        const bool ok_3     = !after_begin || pend == 0;
        const bool ok_sub   = (pend & ref_valid[g]) == 0;

        if (!ok_owner || !ok_1 || !ok_2 || !ok_3 || !ok_sub) {
            LLAMA_LOG_ERROR("%s: turbot invariant violated at granule %u slot %d (owner %d, 1:%d 2:%d 3:%d pending-subset:%d): "
                    "live %016" PRIx64 " ref_valid %016" PRIx64 " pending %016" PRIx64 "\n",
                    where, g, gslot[g], owner[gslot[g]], ok_1, ok_2, ok_3, ok_sub, live, ref_valid[g], pend);
            GGML_ABORT("turbot: tier invariant violated");
        }
    }

    if (n_owned + free_slots.size() != n_slot) {
        LLAMA_LOG_ERROR("%s: turbot invariant violated: %u owned + %zu free slots != %u\n", where, n_owned, free_slots.size(), n_slot);
        GGML_ABORT("turbot: tier invariant violated");
    }
}
