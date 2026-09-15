#include "llama-kv-tier.h"

#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-impl.h"

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

static bool llama_turbot_plan_parse_impl(const std::string & text, const std::string & source,
        const std::vector<int32_t> & attn_layers, uint32_t kv_size, llama_turbot_plan & plan, std::string & err) {
    const auto fail = [&](int line_no, const std::string & msg) {
        err = line_no > 0 ? format("turbot: plan %s line %d: %s", source.c_str(), line_no, msg.c_str())
                          : format("turbot: plan %s: %s", source.c_str(), msg.c_str());
        return false;
    };

    // one L or Y line: the 4 K widths and the 4 V widths of a layer
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
            if (tok.size() != 12 || tok[2] != "K" || tok[7] != "V") {
                return fail(line_no, format("malformed %s line, expected '%s <il> K <w0> <w1> <w2> <w3> V <w0> <w1> <w2> <w3>'",
                        tag.c_str(), tag.c_str()));
            }

            int64_t il = 0;
            if (!llama_turbot_parse_int(tok[1], il) || il < 0 || il > std::numeric_limits<int32_t>::max()) {
                return fail(line_no, "bad layer index '" + tok[1] + "'");
            }

            widths_line wl;
            wl.line_no = line_no;
            for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
                int64_t wk = 0;
                int64_t wv = 0;
                if (!llama_turbot_parse_int(tok[3 + h], wk) || !llama_turbot_parse_int(tok[8 + h], wv)) {
                    return fail(line_no, format("malformed %s line: widths must be integers", tag.c_str()));
                }
                if (is_l) {
                    if (wk < GGML_TURBOT_B_MIN || wk > GGML_TURBOT_B_MAX || wv < GGML_TURBOT_B_MIN || wv > GGML_TURBOT_B_MAX) {
                        return fail(line_no, format("layer %d head %d: old width must be in [%d, %d] (K %lld, V %lld)",
                                (int) il, h, GGML_TURBOT_B_MIN, GGML_TURBOT_B_MAX, (long long) wk, (long long) wv));
                    }
                } else {
                    // the lower bound b+1 depends on the L line and is checked once every line is read
                    if (wk < 1 || wk > GGML_TURBOT_Y_MAX || wv < 1 || wv > GGML_TURBOT_Y_MAX) {
                        return fail(line_no, format("layer %d head %d: young width must be in [b+1, %d] (K %lld, V %lld)",
                                (int) il, h, GGML_TURBOT_Y_MAX, (long long) wk, (long long) wv));
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
    if (pool > (int64_t) kv_size) {
        const int64_t clamped = ((int64_t) kv_size / GGML_TURBOT_GRANULE) * GGML_TURBOT_GRANULE;
        LLAMA_LOG_WARN("%s: turbot plan %s%s: POOL %lld is larger than the cache (%u cells), using %lld\n", __func__, source.c_str(),
                line_pool ? format(" line %d", line_pool).c_str() : "", (long long) pool, kv_size, (long long) clamped);
        pool = clamped;
    }

    const std::set<int32_t> attn(attn_layers.begin(), attn_layers.end());

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

        uint8_t yk[4];
        uint8_t yv[4];
        for (int h = 0; h < GGML_TURBOT_N_HEAD; ++h) {
            yk[h] = ity != lines_y.end() ? ity->second.k[h] : (uint8_t) GGML_TURBOT_Y_DEFAULT;
            yv[h] = ity != lines_y.end() ? ity->second.v[h] : (uint8_t) GGML_TURBOT_Y_DEFAULT;
            if (yk[h] <= wl.k[h] || yk[h] > GGML_TURBOT_Y_MAX || yv[h] <= wl.v[h] || yv[h] > GGML_TURBOT_Y_MAX) {
                return fail(ity != lines_y.end() ? ity->second.line_no : wl.line_no,
                        format("layer %d head %d: young width must be in [b+1, %d] (K b %d y %d, V b %d y %d)",
                            il, h, GGML_TURBOT_Y_MAX, wl.k[h], yk[h], wl.v[h], yv[h]));
            }
        }

        ggml_turbot_layer l;
        if (!ggml_turbot_layer_init(&l, wl.k, wl.v, yk, yv)) {
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

    if (line_bench) {
        LLAMA_LOG_WARN("%s: turbot plan %s line %d: the kvfq bench keys W, W2 and M are ignored\n", __func__, source.c_str(), line_bench);
    }

    plan = std::move(res);
    err.clear();

    return true;
}

bool llama_turbot_plan_parse_text(const std::string & text, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err) {
    return llama_turbot_plan_parse_impl(text, "<text>", attn_layers, kv_size, plan, err);
}

bool llama_turbot_plan_parse_file(const std::string & path, const std::vector<int32_t> & attn_layers, uint32_t kv_size,
                                  llama_turbot_plan & plan, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = format("turbot: cannot open plan file %s", path.c_str());
        return false;
    }

    std::stringstream buf;
    buf << f.rdbuf();

    if (!llama_turbot_plan_parse_impl(buf.str(), path, attn_layers, kv_size, plan, err)) {
        return false;
    }

    plan.path = path;

    return true;
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
