// [TAG_FN_L22_RESSET] [TAG_FN_L22_REWARM] [TAG_FN_L22_SSDTIER] round 22, P5: the residency table (one class per routed
// expert: RAM / VRAM / PIN / SSD), its service thread (standby drop, eviction prefetch, idle re-warm, SSD tier, the
// residency log) and the warm list. See llama-fn-l22-p5.h. Residency and read paths only: nothing changes a value the
// model computes.

#include "llama-fn-l22.h"
#include "llama-fn-l22-p5.h"
#include "llama-fn-res-core.h"

#include "llama-fn-auto.h"
#include "llama-fn-stall.h"
#include "llama-fn-tier.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moecache.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <psapi.h>
#endif

namespace {

using namespace fn_res;

constexpr size_t GIB   = (size_t) 1 << 30;
constexpr size_t BLOCK = (size_t) 2 << 20; // the re-warm's sample and read unit

struct res_layer {
    int                                      il      = -1;
    const ggml_tensor *                      t[3]    = {}; // up, gate, down
    const char *                             base[3] = {};
    size_t                                   nb[3]   = {};
    int                                      n_exp   = 0;
    size_t                                   eb      = 0;  // bytes of one expert (the three slices)
    bool                                     lend_est = false; // in the stream's lend range (estimate at ready)
    std::unique_ptr<std::atomic<uint8_t>[]>  fl;
    std::vector<int32_t>                     slot;
    std::vector<float>                       cnt;
    std::vector<uint8_t>                     vol;
    int32_t                                  b_lo = -1; // [TAG_FN_L22_UBPLAN] P10 R2: the last borrow slots [b_lo, b_hi) seen
    int32_t                                  b_hi = -1;
};

struct res_table {
    const llama_model *     model = nullptr;
    std::vector<res_layer>  L;
    std::vector<int>        by_il;
    int                     n_layer = 0;
    size_t                  page    = 4096;
    bool                    resset  = false;
    bool                    rewarm  = false;
    int                     ssd     = 0;
    bool                    borrow_ram = false; // [TAG_FN_L22_UBPLAN] P10 R2: residents of the 16K reserve's borrow slots stream
    size_t                  floor_b = (size_t) 4096 << 20; // LLAMA_FN_L15_WARM_FREE_MIB
    size_t                  rewarm_floor = (size_t) 6 << 30;
    std::atomic<uint64_t>   gen{1};
    std::mutex              mtx;  // SSD planning and the eviction queue (taken before pmtx)
    std::mutex              pmtx; // the hot set's view: slot / cnt / vol of every layer
    std::vector<std::pair<int, int>> evq;  // left VRAM, to touch when idle
    std::vector<std::pair<int, int>> dropq; // to leave the working set at the next quiet moment (no decode for 50 ms)
    std::thread             svc;
    std::atomic<bool>       stop{false};
    std::atomic<bool>       warm_done{false};
    bool                    lend_known = false; // the lend estimate ran (or nothing lends): the service thread retries until then
    // log counters since the last line
    std::atomic<uint64_t>   n_drop{0};
    std::atomic<uint64_t>   b_drop{0};
    std::atomic<uint64_t>   b_evpf{0};
    std::atomic<uint64_t>   b_rewarm{0};
    std::atomic<uint64_t>   n_ssd_mark{0};
    std::atomic<uint64_t>   n_ssd_unmark{0};
    std::atomic<double>     share{-1.0}; // resident share of the streamed set at the last sample
    // [TAG_FN_L22_SSDFIX] the cache probe (fn_res::probe_*): state 0 none, 1 reading the set in, 2 watching; counters
    std::atomic<int>        probe_state{0};
    std::atomic<uint64_t>   n_probe{0};
    std::atomic<uint64_t>   n_probe_ok{0};
    std::atomic<uint64_t>   n_probe_lost{0};
    std::atomic<uint64_t>   b_probe{0};
    std::mutex              lmtx;        // the log line's interval state
    uint64_t                dio_b0 = 0;
    uint64_t                dio_r0 = 0;
};

std::atomic<res_table *> g_res{nullptr};
std::atomic<int>         g_res_readers{0};

// a reader of the table: the deleter waits until no reader holds it
struct res_ref {
    res_table * t       = nullptr;
    bool        counted = false;
    explicit res_ref(const llama_model * m) {
        if (g_res.load(std::memory_order_relaxed) == nullptr) {
            return; // no table: no shared counter traffic for the callers' threads
        }
        g_res_readers.fetch_add(1);
        counted = true;
        t = g_res.load();
        if (t != nullptr && t->model != m) {
            t = nullptr;
        }
    }
    ~res_ref() {
        if (counted) {
            g_res_readers.fetch_sub(1);
        }
    }
    res_ref(const res_ref &) = delete;
    res_ref & operator=(const res_ref &) = delete;
};

const res_layer * res_find(const res_table * t, int il) {
    if (il < 0 || il >= (int) t->by_il.size() || t->by_il[il] < 0) {
        return nullptr;
    }
    return &t->L[t->by_il[il]];
}

// the flag byte changed by fn (a CAS loop: the planner's PIN marks and the poll's VRAM bits never overwrite each other)
template <typename F>
uint8_t res_update(std::atomic<uint8_t> & a, F fn) {
    uint8_t f = a.load();
    uint8_t nf;
    do {
        nf = fn(f);
    } while (nf != f && !a.compare_exchange_weak(f, nf));
    return f;
}

size_t res_slice_bytes(const res_layer & L) {
    return L.nb[0] + L.nb[1] + L.nb[2];
}

#ifdef _WIN32

size_t fn_avail() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return (size_t) ms.ullAvailPhys;
}

// one Valid bit per address (QueryWorkingSetEx in chunks)
void fn_sample(const std::vector<uintptr_t> & a, std::vector<uint8_t> & valid) {
    valid.assign(a.size(), 0);
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> q;
    const size_t chunk = 65536;
    for (size_t i0 = 0; i0 < a.size(); i0 += chunk) {
        const size_t i1 = std::min(a.size(), i0 + chunk);
        q.assign(i1 - i0, PSAPI_WORKING_SET_EX_INFORMATION{});
        for (size_t i = i0; i < i1; ++i) {
            q[i - i0].VirtualAddress = (PVOID) a[i];
        }
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), q.data(), (DWORD) (q.size()*sizeof(q[0])))) {
            continue; // unknown: counted as missing
        }
        for (size_t i = i0; i < i1; ++i) {
            valid[i] = q[i - i0].VirtualAttributes.Valid ? 1 : 0;
        }
    }
}

using prefetch_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
prefetch_t fn_prefetch_proc() {
    static const prefetch_t pf = (prefetch_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    return pf;
}

void fn_prefetch(const std::vector<range> & rs) {
    const prefetch_t pf = fn_prefetch_proc();
    if (pf == nullptr || rs.empty()) {
        return;
    }
    std::vector<WIN32_MEMORY_RANGE_ENTRY> e;
    for (const auto & r : rs) {
        if (r.size() > 0) {
            e.push_back({ (PVOID) r.lo, (SIZE_T) r.size() });
        }
        if (e.size() == 4096) {
            pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
            e.clear();
        }
    }
    if (!e.empty()) {
        pf(GetCurrentProcess(), (ULONG_PTR) e.size(), e.data(), 0);
    }
}

#endif

// per-expert resident share (0..1): the first page of each slice and one page per 2 MiB inside it
void res_expert_share(const res_table * t, const std::vector<std::pair<int, int>> & ex, std::vector<float> & share) {
    share.assign(ex.size(), 0.0f);
#ifdef _WIN32
    std::vector<uintptr_t> a;
    std::vector<uint32_t>  cnt(ex.size(), 0);
    for (size_t i = 0; i < ex.size(); ++i) {
        const res_layer & L = t->L[ex[i].first];
        for (int k = 0; k < 3; ++k) {
            const range r = align_out({ (uintptr_t) L.base[k] + (uintptr_t) ex[i].second*L.nb[k],
                                        (uintptr_t) L.base[k] + (uintptr_t) (ex[i].second + 1)*L.nb[k] }, t->page);
            for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
                a.push_back(p);
                cnt[i]++;
            }
        }
    }
    std::vector<uint8_t> v;
    fn_sample(a, v);
    size_t j = 0;
    for (size_t i = 0; i < ex.size(); ++i) {
        uint32_t ok = 0;
        for (uint32_t k = 0; k < cnt[i]; ++k) {
            ok += v[j++];
        }
        share[i] = cnt[i] ? (float) ok/cnt[i] : 0.0f;
    }
#else
    GGML_UNUSED(t);
#endif
}

// the pages of these experts leave the working set for the standby list (VirtualUnlock of unlocked pages); whole pages
// inside runs of consecutive listed experts only, so a neighbour's shared edge page stays. Returns the bytes
size_t res_drop(res_table * t, const std::vector<std::pair<int, int>> & ex) {
#ifdef _WIN32
    if (ex.empty()) {
        return 0;
    }
    std::vector<std::vector<uint8_t>> sel(t->L.size());
    for (const auto & x : ex) {
        auto & s = sel[x.first];
        if (s.empty()) {
            s.assign((size_t) t->L[x.first].n_exp, 0);
        }
        s[x.second] = 1;
    }
    size_t b = 0;
    std::vector<range> rs;
    for (size_t li = 0; li < t->L.size(); ++li) {
        if (sel[li].empty()) {
            continue;
        }
        const res_layer & L = t->L[li];
        for (int k = 0; k < 3; ++k) {
            rs.clear();
            runs_of((uintptr_t) L.base[k], L.nb[k], sel[li].data(), L.n_exp, rs);
            for (const auto & r : rs) {
                const range in = align_in(r, t->page);
                if (in.size() > 0) {
                    VirtualUnlock((LPVOID) in.lo, in.size()); // ERROR_NOT_LOCKED: the pages left the working set
                    b += in.size();
                }
            }
        }
    }
    t->n_drop.fetch_add(ex.size());
    t->b_drop.fetch_add(b);
    return b;
#else
    GGML_UNUSED(t);
    GGML_UNUSED(ex);
    return 0;
#endif
}

// the VRAM-held experts that may leave RAM (not pinned), and the SSD-marked ones out of every slot
std::vector<std::pair<int, int>> res_droppable(const res_table * t, bool vram, bool ssd) {
    std::vector<std::pair<int, int>> out;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        const res_layer & L = t->L[li];
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            if (f & F_PIN) {
                continue;
            }
            if ((vram && (f & F_VRAM)) || (ssd && (f & F_SSD) && !(f & F_SLOT))) {
                out.push_back({ li, e });
            }
        }
    }
    return out;
}

// a drop removes pages from the working set, and every processor that runs a thread of this process flushes its TLB:
// the service thread does it only while no decode or prompt runs (queued here)
void res_queue_drop(res_table * t, const std::vector<std::pair<int, int>> & ex) {
    if (ex.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->mtx);
    t->dropq.insert(t->dropq.end(), ex.begin(), ex.end());
    if (t->dropq.size() > 65536) { // a long answer: each expert once
        std::sort(t->dropq.begin(), t->dropq.end());
        t->dropq.erase(std::unique(t->dropq.begin(), t->dropq.end()), t->dropq.end());
    }
}

// no decode or prompt scope runs now nor ran in the last 50 ms
bool res_quiet() {
    int64_t act = 0;
    return !llama_fn_p5_busy(&act) && llama_fn_p5_now_us() - act >= 50000;
}

// [TAG_FN_L22_RESSET] the layers in the prefill stream's lend range, estimated as llama_moe_hot_lend places it
// (llama-moecache.cpp) for the banks pfs_take asks for (llama-prefill-stream.cpp): before the first lend the hot set's
// view does not mark them yet, and their residents would look VRAM-held at ready
bool res_lend_estimate(res_table * t) {
    const char * le = getenv("LLAMA_PREFILL_STREAM_LEND");
    if (le == nullptr || atoi(le) == 0) {
        return true;
    }
    const char * nbs = getenv("LLAMA_PREFILL_STREAM_BUFS");
    const int n_bufs = std::max(1, std::min(3, nbs && nbs[0] ? atoi(nbs) : 2));
    ggml_backend_buffer_t dbuf = nullptr;
    std::vector<const llama_moe_cache_layer *> cl(t->L.size(), nullptr);
    for (size_t li = 0; li < t->L.size(); ++li) {
        const llama_moe_cache_layer * c = llama_moe_cache_lookup_table(t->L[li].t[0]);
        if (c == nullptr || !c->up_c || !c->gate_c || !c->down_c) {
            continue;
        }
        if (dbuf == nullptr) {
            dbuf = c->up_c->buffer;
        }
        if (c->up_c->buffer != dbuf || c->gate_c->buffer != dbuf || c->down_c->buffer != dbuf) {
            return true; // not one device buffer: the stream does not lend
        }
        cl[li] = c;
    }
    if (dbuf == nullptr) {
        return false; // no hot set yet
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dbuf);
    // the stream sizes its banks with the buffer type of the first host layer's router (llama-moe-gen5.cpp: d.buft)
    const ggml_tensor *        router = t->model->layers[t->L[0].il].ffn_gate_inp;
    ggml_backend_buffer_type_t sbuft  = router && router->buffer ? ggml_backend_buffer_get_type(router->buffer) : buft;
    const size_t balign = std::max<size_t>(256, ggml_backend_buft_get_alignment(sbuft));
    size_t mx[3] = { 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int k = 0; k < 3; ++k) {
            mx[k] = std::max(mx[k], ggml_backend_buft_get_alloc_size(sbuft, L.t[k]));
        }
    }
    const size_t bank  = align_up(mx[0], balign) + align_up(mx[1], balign) + align_up(mx[2], balign);
    const size_t size  = ggml_backend_buffer_get_size(dbuf);
    const size_t align = std::max<size_t>(256, ggml_backend_buffer_get_alignment(dbuf));
    size_t lo = 0;
    size_t hi = 0;
    if (!lend_range(size, llama_moe_hot_cbuf_tail_bytes(), (size_t) n_bufs*bank, align, &lo, &hi)) {
        return true;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(dbuf);
    int n = 0;
    for (size_t li = 0; li < t->L.size(); ++li) {
        if (cl[li] == nullptr) {
            continue;
        }
        for (const ggml_tensor * x : { cl[li]->up_c, cl[li]->gate_c, cl[li]->down_c }) {
            const uintptr_t a = (uintptr_t) x->data;
            if (overlaps(a, a + ggml_backend_buft_get_alloc_size(buft, x), base + lo, base + hi)) {
                t->L[li].lend_est = true;
            }
        }
        n += t->L[li].lend_est ? 1 : 0;
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] %d layers lie in the prefill stream's lend range (%.0f MiB, estimate): their "
            "residents count as streamed\n", __func__, n, (hi - lo)/1048576.0);
    return true;
}

// one layer's view from the hot set; entered / left: experts that came into / left a VRAM-held slot. false: unchanged
bool res_poll_layer(res_table * t, int li, std::vector<std::pair<int, int>> * entered, std::vector<std::pair<int, int>> * left) {
    res_layer & L = t->L[li];
    std::vector<int32_t> s((size_t) L.n_exp, -1);
    std::vector<float>   c((size_t) L.n_exp, 0.0f);
    std::vector<uint8_t> v((size_t) L.n_exp, 0);
    const int r = llama_moe_hot_tier_view(L.t[0], s.data(), c.data(), v.data(), L.n_exp);
    if (r == 0) {
        return false; // busy right now: the last view stays
    }
    if (r < 0) {
        std::fill(s.begin(), s.end(), -1); // no hot set (yet): every expert is the host's
        std::fill(v.begin(), v.end(), 0);
    }
    // [TAG_FN_L22_UBPLAN] [TAG_FN_L22_BIGUB2] P10 R2: with UBPLAN / BIGUB2 the borrow slots go out in every 16K FULL period
    // and the stream reads their experts from RAM then: they lose their VRAM without a choice (vol), so RAM class. A busy
    // query keeps the last range (a stale one only keeps pages); no borrow anywhere or no hot set clears it.
    if (t->borrow_ram) {
        int32_t lo = -1, hi = -1;
        if (r > 0 && llama_moe_hot_borrow_range(L.t[0], &lo, &hi)) {
            L.b_lo = lo;
            L.b_hi = hi;
        } else if (r < 0 || llama_moe_hot_cb_borrow_bytes() == 0) {
            L.b_lo = L.b_hi = -1;
        }
        for (int e = 0; e < L.n_exp && L.b_lo >= 0; ++e) {
            v[e] = s[e] >= L.b_lo && s[e] < L.b_hi ? 1 : v[e];
        }
    }
    bool changed = false;
    for (int e = 0; e < L.n_exp; ++e) {
        const bool in_slot = s[e] >= 0;
        const bool vram    = in_slot && !v[e] && !L.lend_est;
        const uint8_t f = res_update(L.fl[e], [&](uint8_t x) {
            uint8_t n = (uint8_t) (x & ~(F_VRAM | F_SLOT));
            n |= in_slot ? F_SLOT : 0;
            n |= vram ? F_VRAM : 0;
            if (in_slot) {
                n &= (uint8_t) ~F_SSD; // a hot expert is never the least routed
            }
            return n;
        });
        const bool was = (f & F_VRAM) != 0;
        if (vram && !was && entered) {
            entered->push_back({ li, e });
        }
        if (!vram && was && left) {
            left->push_back({ li, e });
        }
        changed = changed || was != vram || ((f & F_SLOT) != 0) != in_slot || (in_slot && (f & F_SSD));
    }
    L.slot.swap(s);
    L.cnt.swap(c);
    L.vol.swap(v);
    return changed;
}

bool res_poll(res_table * t, std::vector<std::pair<int, int>> * entered, std::vector<std::pair<int, int>> * left) {
    std::lock_guard<std::mutex> lk(t->pmtx);
    bool changed = false;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        changed = res_poll_layer(t, li, entered, left) || changed;
    }
    if (changed) {
        t->gen.fetch_add(1);
    }
    return changed;
}

// experts of the streamed set (not VRAM-held, not pinned); with_ssd: the SSD-marked ones too
std::vector<std::pair<int, int>> res_streamed(const res_table * t, bool with_ssd) {
    std::vector<std::pair<int, int>> out;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        const res_layer & L = t->L[li];
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            if ((f & (F_VRAM | F_PIN)) || (!with_ssd && (f & F_SSD))) {
                continue;
            }
            out.push_back({ li, e });
        }
    }
    return out;
}

// [TAG_FN_L22_SSDTIER] SSD marks for the least-routed cold experts of the streamed set until it fits in RAM.
// stage 0 (ready, before the warm pass): the whole set against resident + room, resident picks go to standby;
// stage 1 (after the warm pass, and every 60 s when idle): the non-resident part against the room; stage 1 also gives
// marks back (most routed first) when more than 1 GiB of room is left. Under t->mtx
void res_ssd_plan(res_table * t, int stage, size_t lw_room = SIZE_MAX) {
#ifdef _WIN32
    if (t->ssd < 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->mtx);
    std::lock_guard<std::mutex> plk(t->pmtx); // the counts
    const auto S = res_streamed(t, stage == 0);
    std::vector<float> sh;
    res_expert_share(t, S, sh);
    size_t need = 0;
    size_t res  = 0;
    for (size_t i = 0; i < S.size(); ++i) {
        const size_t b = t->L[S[i].first].eb;
        need += b;
        res  += (size_t) (b*sh[i]);
    }
    // pinned experts not in RAM yet (the driver brings them in as it registers them)
    std::vector<std::pair<int, int>> P;
    for (int li = 0; li < (int) t->L.size(); ++li) {
        for (int e = 0; e < t->L[li].n_exp; ++e) {
            if (t->L[li].fl[e].load() & F_PIN) {
                P.push_back({ li, e });
            }
        }
    }
    std::vector<float> psh;
    res_expert_share(t, P, psh);
    size_t pin_pending = 0;
    for (size_t i = 0; i < P.size(); ++i) {
        pin_pending += (size_t) (t->L[P[i].first].eb*(1.0f - psh[i]));
    }
    const size_t avail = fn_avail();
    std::vector<ssd_cand> c;
    std::vector<size_t>   ci; // index into S
    size_t deficit = 0;
    if (stage == 0) {
        deficit = ssd_deficit(need, res, avail, t->floor_b, pin_pending);
    } else {
        const size_t nonres = need - std::min(need, res);
        const size_t room   = avail > t->floor_b + pin_pending ? avail - t->floor_b - pin_pending : 0;
        deficit = nonres > room ? nonres - room : 0;
        if (deficit == 0 && room > nonres + GIB) {
            // marks back while the room allows (the idle re-warm reads them)
            std::vector<ssd_cand> m;
            std::vector<std::pair<int, int>> mx;
            for (int li = 0; li < (int) t->L.size(); ++li) {
                const res_layer & L = t->L[li];
                for (int e = 0; e < L.n_exp; ++e) {
                    const uint8_t f = L.fl[e].load();
                    if ((f & F_SSD) && !(f & (F_PIN | F_SLOT))) {
                        m.push_back({ li, e, e < (int) L.cnt.size() ? L.cnt[e] : 0.0f, L.eb, false });
                        mx.push_back({ li, e });
                    }
                }
            }
            // [TAG_FN_L22_SSDFIX] no more than the prompts' lowest available RAM leaves room for (lw_room)
            const auto un = ssd_unpick(m, std::min(room - nonres - GIB, lw_room));
            for (size_t i : un) {
                res_update(t->L[mx[i].first].fl[mx[i].second], [](uint8_t x) { return (uint8_t) (x & ~F_SSD); });
            }
            if (!un.empty()) {
                t->n_ssd_unmark.fetch_add(un.size());
                t->gen.fetch_add(1);
            }
            return;
        }
    }
    if (deficit == 0) {
        return;
    }
    for (size_t i = 0; i < S.size(); ++i) {
        const res_layer & L = t->L[S[i].first];
        const int e = S[i].second;
        const uint8_t f = L.fl[e].load();
        if ((f & (F_SLOT | F_SSD | F_PIN)) || (stage == 1 && sh[i] >= 0.5f)) {
            continue; // only cold experts; after the warm pass only the ones not in RAM
        }
        c.push_back({ S[i].first, e, e < (int) L.cnt.size() ? L.cnt[e] : 0.0f, L.eb, sh[i] >= 0.5f });
        ci.push_back(i);
    }
    const auto pick = ssd_pick(c, deficit, true);
    std::vector<std::pair<int, int>> drop;
    size_t marked = 0;
    for (size_t k : pick) {
        const auto & x = S[ci[k]];
        bool set = false;
        res_update(t->L[x.first].fl[x.second], [&](uint8_t f) {
            set = !(f & (F_SLOT | F_PIN));
            return set ? (uint8_t) (f | F_SSD) : f;
        });
        if (set) {
            marked += t->L[x.first].eb;
            if (c[k].resident) {
                drop.push_back(x);
            }
        }
    }
    if (marked > 0) {
        t->n_ssd_mark.fetch_add(pick.size());
        t->gen.fetch_add(1);
    }
    size_t dropped = 0;
    for (const auto & x : drop) {
        dropped += t->L[x.first].eb;
    }
    if (res_quiet()) {
        res_drop(t, drop); // now: the room it frees counts for the warm pass and the next plan
    } else {
        t->dropq.insert(t->dropq.end(), drop.begin(), drop.end()); // t->mtx is held
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDTIER] %s: the streamed set needs %.2f GiB, %.2f GiB in RAM, %.2f GiB available (floor %.1f, "
            "pins pending %.2f): %.2f GiB of %zu least-routed cold experts are read by direct I/O in prompts (%.2f GiB of them "
            "leave the working set)\n", __func__, stage == 0 ? "before the warm pass" : "what does not fit", need/1073741824.0,
            res/1073741824.0, avail/1073741824.0, t->floor_b/1073741824.0, pin_pending/1073741824.0, marked/1073741824.0,
            pick.size(), dropped/1073741824.0);
#else
    GGML_UNUSED(t);
    GGML_UNUSED(stage);
#endif
}

// [TAG_FN_L22_REWARM] the streamed set's missing 2 MiB blocks, in stream order (layer order); *share: resident share;
// *miss_b (optional): their bytes
void res_missing_blocks(res_table * t, std::vector<range> & miss, double * share, size_t * miss_b = nullptr) {
    miss.clear();
    if (miss_b) {
        *miss_b = 0;
    }
#ifdef _WIN32
    std::vector<range> runs;
    std::vector<uint8_t> sel;
    for (const auto & L : t->L) {
        sel.assign((size_t) L.n_exp, 0);
        for (int e = 0; e < L.n_exp; ++e) {
            const uint8_t f = L.fl[e].load();
            sel[e] = (f & (F_VRAM | F_PIN | F_SSD)) ? 0 : 1;
        }
        for (int k = 0; k < 3; ++k) {
            const size_t r0 = runs.size();
            runs_of((uintptr_t) L.base[k], L.nb[k], sel.data(), L.n_exp, runs);
            for (size_t i = r0; i < runs.size(); ++i) {
                runs[i] = align_out(runs[i], t->page);
            }
        }
    }
    std::vector<range>     blocks;
    std::vector<uintptr_t> a;
    for (const auto & r : runs) {
        for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
            blocks.push_back({ p, std::min<uintptr_t>(r.hi, p + BLOCK) });
            a.push_back(p);
        }
    }
    std::vector<uint8_t> v;
    fn_sample(a, v);
    size_t tot = 0;
    size_t ok  = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        tot += blocks[i].size();
        if (v[i]) {
            ok += blocks[i].size();
        } else {
            miss.push_back(blocks[i]);
        }
    }
    *share = tot ? (double) ok/tot : 1.0;
    if (miss_b) {
        *miss_b = tot - ok;
    }
#else
    GGML_UNUSED(t);
    *share = 1.0;
#endif
}

// touch these ranges at low priority until activity: false when it stopped early (*done: bytes touched)
bool res_touch(const std::vector<range> & rs, size_t page, int64_t act0, size_t * done) {
#ifdef _WIN32
    *done = 0;
    for (const auto & r : rs) {
        for (uintptr_t p = r.lo; p < r.hi; p += BLOCK) {
            int64_t act = 0;
            if (llama_fn_p5_busy(&act) || act != act0) {
                return false;
            }
            const uintptr_t q = std::min<uintptr_t>(r.hi, p + BLOCK);
            for (uintptr_t x = p; x < q; x += page) {
                (void) *(volatile const char *) x;
            }
            *done += (size_t) (q - p);
        }
    }
    return true;
#else
    GGML_UNUSED(rs);
    GGML_UNUSED(page);
    GGML_UNUSED(act0);
    *done = 0;
    return true;
#endif
}

std::vector<range> res_expert_ranges(const res_table * t, const std::vector<std::pair<int, int>> & ex) {
    std::vector<range> rs;
    for (const auto & x : ex) {
        const res_layer & L = t->L[x.first];
        for (int k = 0; k < 3; ++k) {
            rs.push_back(align_out({ (uintptr_t) L.base[k] + (uintptr_t) x.second*L.nb[k],
                                     (uintptr_t) L.base[k] + (uintptr_t) (x.second + 1)*L.nb[k] }, t->page));
        }
    }
    return rs;
}

void res_log(res_table * t, const char * when) {
    if (llama_fn_l22_passlog_level() < 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(t->lmtx);
    size_t cls[4] = { 0, 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            cls[class_of(L.fl[e].load())] += L.eb;
        }
    }
    uint64_t db = 0;
    uint64_t dr = 0;
    uint64_t df = 0;
    llama_fn_p5_dio_stats(&db, &dr, &df);
#ifdef _WIN32
    const size_t avail = fn_avail();
#else
    const size_t avail = 0;
#endif
    llama_fn_l22_log("res", "when=%s avail_gib=%.2f streamed_gib=%.2f resident_share=%.3f vram_gib=%.2f pin_gib=%.2f ssd_gib=%.2f "
            "drops=%" PRIu64 " drop_gib=%.2f evict_pf_gib=%.2f rewarm_gib=%.2f ssd_marks=%" PRIu64 " ssd_unmarks=%" PRIu64
            " dio_gib=%.2f dio_reads=%" PRIu64 " dio_fails=%" PRIu64 " gen=%" PRIu64 " probe=%d probes=%" PRIu64 " probe_kept=%"
            PRIu64 " probe_lost=%" PRIu64 " probe_gib=%.2f", when, avail/1073741824.0,
            (cls[0] + cls[3])/1073741824.0, t->share.load(), cls[1]/1073741824.0, cls[2]/1073741824.0, cls[3]/1073741824.0,
            t->n_drop.exchange(0), t->b_drop.exchange(0)/1073741824.0, t->b_evpf.exchange(0)/1073741824.0,
            t->b_rewarm.exchange(0)/1073741824.0, t->n_ssd_mark.exchange(0), t->n_ssd_unmark.exchange(0),
            (db - t->dio_b0)/1073741824.0, dr - t->dio_r0, df, t->gen.load(), t->probe_state.load(), t->n_probe.load(),
            t->n_probe_ok.load(), t->n_probe_lost.load(), t->b_probe.exchange(0)/1073741824.0);
    t->dio_b0 = db;
    t->dio_r0 = dr;
}

// [TAG_FN_L22_SSDFIX] the SSD tier's cache probe (fn_res::probe_fits / probe_verdict / probe_backoff), one step per idle
// tick of the service thread. The marked set is read through the mapping when it fits above PROBE_FREE of available
// RAM: unmarked (the stream reads it mapped from then on), prefetched and touched at low priority (stops at any
// activity), then watched: kept when after a prompt or answer the streamed set's missing bytes did not grow; lost when
// they did (the plan marks the least-routed missing experts again, the next probe waits twice as long, up to 32 min)
constexpr int64_t PROBE_FIRST_US = 60000000;          // the first probe a minute after the warm pass; kept: the next look
constexpr int64_t PROBE_BASE_US  = 120000000;         // the wait after a verdict
constexpr int64_t PROBE_MAX_US   = 32*60000000LL;
constexpr int64_t PROBE_WATCH_US = 20000000;          // the watch lasts at least this long (and one prompt or answer)
constexpr size_t  PROBE_FREE     = (size_t) 2 << 30;  // a probe never takes the available RAM below this
constexpr size_t  PROBE_CUSHION  = (size_t) 3 << 30;  // nor the lowest available RAM seen in prompts below this (pinned reserves, store)
constexpr size_t  PROBE_WATCH_MIN = (size_t) 2 << 30; // a watch that sees less loses
constexpr int64_t PROBE_LW_US    = 60000000;          // the low-water window (two of them are kept)

struct probe_ctx {
    int64_t            next   = 0; // 0: armed at the first idle tick after the warm pass
    int64_t            wait   = PROBE_BASE_US;
    int64_t            t0     = 0;
    int64_t            t_samp = 0;
    int64_t            act0   = 0;
    size_t             miss0  = 0;
    size_t             probed = 0;
    bool               listed = false;
    std::vector<range> miss;
    size_t             at     = 0;
    // the lowest available RAM over the last two windows of PROBE_LW_US, and whether a prompt or answer ran in them: a
    // prompt's FULL reserve takes up to 2.4 GiB of pinned host memory for a moment (16K width), the conversation store
    // more, so the room is what is left at the worst moment, not at the idle tick
    size_t             lw_cur   = SIZE_MAX;
    size_t             lw_prev  = SIZE_MAX;
    bool               act_cur  = false;
    bool               act_prev = false;
    int64_t            lw_t0    = 0;
    uint64_t           n_skip   = 0;
    bool               lw_reset = false;            // the windows restarted after a give-back: no marks back until one is full
    size_t             lw_watch = SIZE_MAX;         // the lowest available RAM since the probe began
    std::vector<std::pair<int, int>> set;           // the probed experts (back to the SSD set when the probe is lost)
};

void res_probe_sample(probe_ctx & p, int64_t now, bool busy) {
#ifdef _WIN32
    const size_t a = fn_avail();
    p.lw_cur   = std::min(p.lw_cur, a);
    p.act_cur  = p.act_cur || busy;
    if (!p.set.empty()) {
        p.lw_watch = std::min(p.lw_watch, a); // a probe runs: its own low-water
    }
    if (p.lw_t0 == 0) {
        p.lw_t0 = now;
    } else if (now - p.lw_t0 >= PROBE_LW_US) {
        p.lw_prev  = p.lw_cur;
        p.act_prev = p.act_cur;
        p.lw_reset = false;
        p.lw_cur   = SIZE_MAX;
        p.act_cur  = false;
        p.lw_t0    = now;
    }
#else
    GGML_UNUSED(p);
    GGML_UNUSED(now);
    GGML_UNUSED(busy);
#endif
}

void res_probe_end(res_table * t, probe_ctx & p, int64_t now, bool kept, size_t miss1, const char * why) {
    t->probe_state.store(0);
    p.wait = probe_backoff(p.wait, kept, PROBE_BASE_US, PROBE_MAX_US);
    p.next = now + (kept ? PROBE_FIRST_US : p.wait);
    p.miss.clear();
    p.at     = 0;
    p.listed = false;
    (kept ? t->n_probe_ok : t->n_probe_lost).fetch_add(1);
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDFIX] cache probe %" PRIu64 " %s (%s): the streamed set misses %.2f GiB (%.2f GiB at the "
            "watch's start, %.2f GiB probed, available RAM at the lowest %.2f GiB)%s; next probe in %.0f s\n", __func__,
            t->n_probe.load(), kept ? "kept" : "lost", why, miss1/1073741824.0, p.miss0/1073741824.0, p.probed/1073741824.0,
            (p.lw_watch == SIZE_MAX ? 0 : p.lw_watch)/1073741824.0,
            kept ? ": its experts are read through the mapping" : ": the set goes back to direct I/O", (p.next - now)/1e6);
    if (!kept) {
        // the probed experts back to the SSD set (not the ones a hot slot or a pin took meanwhile), their pages to the
        // standby list at the next quiet moment (available again at once); then the plan for what else is missing
        std::vector<std::pair<int, int>> back;
        {
            std::lock_guard<std::mutex> lk(t->mtx);
            for (const auto & x : p.set) {
                bool set = false;
                res_update(t->L[x.first].fl[x.second], [&](uint8_t f) {
                    set = !(f & (F_SLOT | F_PIN | F_VRAM));
                    return set ? (uint8_t) (f | F_SSD) : f;
                });
                if (set) {
                    back.push_back(x);
                }
            }
        }
        if (!back.empty()) {
            t->n_ssd_mark.fetch_add(back.size());
            t->gen.fetch_add(1);
            res_queue_drop(t, back);
        }
        res_ssd_plan(t, 1);
    }
    p.set.clear();
    p.lw_watch = SIZE_MAX;
}

void res_probe_step(res_table * t, int64_t now, int64_t act, probe_ctx & p) {
#ifdef _WIN32
    const int st = t->probe_state.load();
    if (st == 0) {
        if (p.next == 0) {
            p.next = now + PROBE_FIRST_US;
            return;
        }
        if (now < p.next) {
            return;
        }
        std::lock_guard<std::mutex> lk(t->mtx); // no plan meanwhile
        std::vector<std::pair<int, int>> m;
        size_t mb = 0;
        for (int li = 0; li < (int) t->L.size(); ++li) {
            const res_layer & L = t->L[li];
            for (int e = 0; e < L.n_exp; ++e) {
                const uint8_t f = L.fl[e].load();
                if ((f & F_SSD) && !(f & (F_SLOT | F_PIN))) {
                    m.push_back({ li, e });
                    mb += L.eb;
                }
            }
        }
        if (m.empty()) {
            p.next = now + PROBE_FIRST_US; // nothing marked: look again later
            return;
        }
        const size_t avail = fn_avail();
        const size_t lw    = std::min(p.lw_cur, p.lw_prev);
        if (p.lw_prev == SIZE_MAX || !(p.act_prev || p.act_cur)) {
            p.next = now + PROBE_FIRST_US; // no full window with a prompt in it yet: its low-water is not known
            return;
        }
        if (!probe_fits(mb, avail, std::max(PROBE_FREE, t->floor_b)) || !probe_fits(mb, lw, PROBE_CUSHION)) {
            p.next = now + p.wait; // no room for it now
            if (++p.n_skip <= 3 || p.n_skip % 16 == 0) {
                LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDFIX] cache probe not taken (%" PRIu64 "): the %.2f GiB SSD set does not fit "
                        "(%.2f GiB available, %.2f GiB at the lowest in the last prompts; it keeps %.1f / %.1f GiB): it stays on "
                        "direct I/O\n", __func__, p.n_skip, mb/1073741824.0, avail/1073741824.0, lw/1073741824.0,
                        std::max(PROBE_FREE, t->floor_b)/1073741824.0, PROBE_CUSHION/1073741824.0);
            }
            return;
        }
        for (const auto & x : m) {
            res_update(t->L[x.first].fl[x.second], [](uint8_t f) { return (uint8_t) (f & ~F_SSD); });
        }
        t->n_ssd_unmark.fetch_add(m.size());
        t->gen.fetch_add(1);
        t->n_probe.fetch_add(1);
        p.probed   = mb;
        p.listed   = false;
        p.at       = 0;
        p.set      = m;
        p.lw_watch = SIZE_MAX;
        t->probe_state.store(1);
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDFIX] cache probe %" PRIu64 ": the %.2f GiB of %zu SSD-marked experts are read through the "
                "mapping now (%.2f GiB available, the probe keeps %.1f GiB free)\n", __func__, t->n_probe.load(), mb/1073741824.0,
                m.size(), avail/1073741824.0, std::max(PROBE_FREE, t->floor_b)/1073741824.0);
        return;
    }
    if (st == 1) {
        if (!p.listed) {
            double sh = 1.0;
            res_missing_blocks(t, p.miss, &sh);
            p.listed = true;
            p.at     = 0;
        }
        if (p.at < p.miss.size()) {
            const size_t avail = fn_avail();
            const size_t keep  = std::max(PROBE_FREE, t->floor_b);
            const size_t room  = avail > keep ? avail - keep : 0;
            std::vector<range> win;
            size_t b = 0;
            while (p.at + win.size() < p.miss.size() && b < ((size_t) 256 << 20) && b + p.miss[p.at + win.size()].size() <= room) {
                b += p.miss[p.at + win.size()].size();
                win.push_back(p.miss[p.at + win.size()]);
            }
            if (win.empty()) {
                size_t mb = 0;
                double sh = 1.0;
                std::vector<range> tmp;
                res_missing_blocks(t, tmp, &sh, &mb);
                p.miss0 = mb;
                res_probe_end(t, p, now, false, mb, "no room left while reading it in");
                return;
            }
            fn_prefetch(win);
            const int64_t t_end = now + 500000;
            size_t i = 0;
            for (; i < win.size() && llama_fn_p5_now_us() < t_end && !t->stop.load(); ++i) {
                size_t done = 0;
                const bool all = res_touch({ win[i] }, t->page, act, &done);
                t->b_probe.fetch_add(done);
                if (!all) {
                    break; // activity: this block again at the next idle tick
                }
            }
            p.at += i;
            return;
        }
        double sh = 1.0;
        std::vector<range> tmp;
        res_missing_blocks(t, tmp, &sh, &p.miss0);
        p.miss.clear();
        p.t0     = now;
        p.t_samp = now;
        p.act0   = act;
        t->probe_state.store(2);
        return;
    }
    // st == 2: the watch, one sample per 5 s of idle time
    if (now - p.t_samp < 5000000) {
        return;
    }
    p.t_samp = now;
    double sh = 1.0;
    size_t mb = 0;
    std::vector<range> tmp;
    res_missing_blocks(t, tmp, &sh, &mb);
    t->share.store(sh);
    // the set took the room a prompt's pinned reserve or the conversation store needs: lost as well
    const bool tight = p.lw_watch != SIZE_MAX && p.lw_watch < PROBE_WATCH_MIN;
    const int  v     = tight ? -1 : probe_verdict(p.miss0, mb, p.probed, act != p.act0, now - p.t0, PROBE_WATCH_US);
    if (v != 0) {
        res_probe_end(t, p, now, v > 0, mb, v > 0 ? "a prompt or answer later" :
                tight ? "available RAM fell below the cushion" : "pages left RAM");
    }
#else
    GGML_UNUSED(t);
    GGML_UNUSED(now);
    GGML_UNUSED(act);
    GGML_UNUSED(p);
#endif
}

// [TAG_FN_L22_SSDFIX] the give-back. The plan sizes the SSD set at an idle tick, but a prompt's FULL reserve takes up to
// 2.4 GiB of pinned host memory for a moment (16K width) and the conversation store more: when the prompts' lowest
// available RAM falls below GIVE_CUSHION, Windows trimmed working sets to find pages (any page: the MTP head's
// experts, a layer the stream reads next) and the prompt stalled for seconds (2-8 s at a prompt's end under the 8 GiB
// eater). The least-routed cold experts that are in RAM then join the SSD set until GIVE_TARGET would be left at that
// moment; their pages go to standby (available at once), the low-water windows start again and the probe waits
constexpr size_t GIVE_CUSHION = (size_t) 2 << 30;
constexpr size_t GIVE_TARGET  = (size_t) 3 << 30;

size_t res_give_back(res_table * t, size_t lw, probe_ctx & p, int64_t now) {
#ifdef _WIN32
    const size_t need = lw < GIVE_TARGET ? GIVE_TARGET - lw : 0;
    if (need == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(t->mtx);
    std::lock_guard<std::mutex> plk(t->pmtx); // the counts
    const auto S = res_streamed(t, false);
    std::vector<float> sh;
    res_expert_share(t, S, sh);
    std::vector<ssd_cand> c;
    std::vector<size_t>   ci;
    for (size_t i = 0; i < S.size(); ++i) {
        const res_layer & L = t->L[S[i].first];
        const int e = S[i].second;
        const uint8_t f = L.fl[e].load();
        if ((f & (F_SLOT | F_SSD | F_PIN)) || sh[i] < 0.5f) {
            continue; // cold experts in RAM only (the missing ones free nothing)
        }
        c.push_back({ S[i].first, e, e < (int) L.cnt.size() ? L.cnt[e] : 0.0f, L.eb, true });
        ci.push_back(i);
    }
    const auto pick = ssd_pick(c, need, false);
    std::vector<std::pair<int, int>> drop;
    size_t marked = 0;
    for (size_t k : pick) {
        const auto & x = S[ci[k]];
        bool set = false;
        res_update(t->L[x.first].fl[x.second], [&](uint8_t f) {
            set = !(f & (F_SLOT | F_PIN));
            return set ? (uint8_t) (f | F_SSD) : f;
        });
        if (set) {
            drop.push_back(x);
            marked += t->L[x.first].eb;
        }
    }
    if (!drop.empty()) {
        t->n_ssd_mark.fetch_add(drop.size());
        t->gen.fetch_add(1);
        if (res_quiet()) {
            res_drop(t, drop);
        } else {
            t->dropq.insert(t->dropq.end(), drop.begin(), drop.end()); // t->mtx is held
        }
    }
    p.lw_cur   = SIZE_MAX;
    p.lw_prev  = SIZE_MAX;
    p.act_cur  = false;
    p.act_prev = false;
    p.lw_t0    = now;
    p.lw_reset = true;
    p.next     = std::max(p.next, now + p.wait);
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_SSDFIX] give-back: available RAM fell to %.2f GiB in the last prompts (cushion %.1f GiB): "
            "%.2f GiB of %zu least-routed cold experts in RAM go to direct I/O, their pages to standby\n", __func__,
            lw/1073741824.0, GIVE_CUSHION/1073741824.0, marked/1073741824.0, drop.size());
    return marked;
#else
    GGML_UNUSED(t);
    GGML_UNUSED(lw);
    GGML_UNUSED(p);
    GGML_UNUSED(now);
    return 0;
#endif
}

// the service thread: the hot set's view every second (drops, eviction prefetch), the idle work (eviction touch,
// re-warm, the 60 s re-drop and SSD plan), the residency log every 60 s
void res_service(res_table * t) {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    llama_fn_thread_name("res");
    int64_t t_poll = 0;
    int64_t t_log  = llama_fn_p5_now_us();
    int64_t t_per  = t_log;
    int64_t t_samp = 0;
    int64_t act_samp = -1;
    uint64_t gen_samp = 0;
    std::vector<range> miss;
    size_t miss_at = 0;
    std::vector<std::pair<int, int>> entered;
    std::vector<std::pair<int, int>> left;
    probe_ctx probe; // [TAG_FN_L22_SSDFIX]
    while (!t->stop.load()) {
        Sleep(100);
        const int64_t now = llama_fn_p5_now_us();
        if (now - t_poll >= 1000000) {
            t_poll = now;
            entered.clear();
            left.clear();
            if (!t->lend_known) {
                t->lend_known = res_lend_estimate(t); // the hot set came after the table
            }
            res_poll(t, &entered, &left);
            if (t->resset) {
                // drop the host copies of experts that came into VRAM (queued); prefetch the ones that left it
                res_queue_drop(t, entered);
                if (!left.empty()) {
                    const auto rs = res_expert_ranges(t, left);
                    fn_prefetch(rs);
                    std::lock_guard<std::mutex> lk(t->mtx);
                    t->evq.insert(t->evq.end(), left.begin(), left.end());
                    if (t->evq.size() > 65536) { // never idle for long: the oldest go (they were prefetched)
                        t->evq.erase(t->evq.begin(), t->evq.begin() + (t->evq.size() - 32768));
                    }
                }
            }
        }
        if (now - t_log >= 60000000) {
            t_log = now;
            res_log(t, "60s");
        }
        int64_t act = 0;
        const bool busy = llama_fn_p5_busy(&act);
        if (t->ssd >= 1) {
            res_probe_sample(probe, now, busy); // [TAG_FN_L22_SSDFIX] the available RAM's low-water, every 100 ms
        }
        if (!busy && now - act >= 50000) {
            std::vector<std::pair<int, int>> q;
            {
                std::lock_guard<std::mutex> lk(t->mtx);
                q.swap(t->dropq);
            }
            // pinned meanwhile, or no longer VRAM-held / SSD (the eviction prefetch wants it back): skip
            q.erase(std::remove_if(q.begin(), q.end(), [&](const std::pair<int, int> & x) {
                const uint8_t f = t->L[x.first].fl[x.second].load();
                return (f & F_PIN) || !(f & (F_VRAM | F_SSD));
            }), q.end());
            res_drop(t, q);
        }
        const bool idle = !busy && now - act >= 1000000 && !llama_fn_warm_running() && t->warm_done.load();
        if (!idle) {
            continue;
        }
        // eviction touches first (small)
        if (t->resset) {
            std::vector<std::pair<int, int>> q;
            {
                std::lock_guard<std::mutex> lk(t->mtx);
                q.swap(t->evq);
            }
            if (!q.empty()) {
                size_t done = 0;
                const auto rs = res_expert_ranges(t, q);
                if (!res_touch(rs, t->page, act, &done)) {
                    std::lock_guard<std::mutex> lk(t->mtx); // stopped by activity: the rest later
                    t->evq.insert(t->evq.begin(), q.begin(), q.end());
                }
                t->b_evpf.fetch_add(done);
            }
        }
        // [TAG_FN_L22_SSDFIX] the SSD set's cache probe (reading it in replaces the re-warm while it runs)
        if (t->ssd >= 1) {
            res_probe_step(t, now, act, probe);
            // the give-back at the first idle moment after prompts took the available RAM below the cushion (the 60 s
            // tick may come minutes later: the first prompts after a start are the tightest)
            if (t->probe_state.load() == 0 && probe.act_cur && probe.lw_cur < GIVE_CUSHION) {
                res_give_back(t, probe.lw_cur, probe, now);
            }
        }
        if (t->rewarm && t->probe_state.load() != 1) {
            // a new sample at each idle period, after a table change, and every 10 s of a long one
            if (act != act_samp || gen_samp != t->gen.load() || now - t_samp >= 10000000) {
                double sh = 1.0;
                res_missing_blocks(t, miss, &sh);
                t->share.store(sh);
                miss_at  = 0;
                t_samp   = now;
                act_samp = act;
                gen_samp = t->gen.load();
            }
            if (miss_at < miss.size()) {
                const size_t avail = fn_avail();
                const size_t room = avail > t->rewarm_floor ? avail - t->rewarm_floor : 0;
                // a window of up to 256 MiB per tick: one prefetch for all of it (the reads overlap), then the touches in
                // order, for at most 500 ms, stopped by any activity
                std::vector<range> win;
                size_t b = 0;
                while (miss_at + win.size() < miss.size() && b < ((size_t) 256 << 20) &&
                       b + miss[miss_at + win.size()].size() <= room) {
                    b += miss[miss_at + win.size()].size();
                    win.push_back(miss[miss_at + win.size()]);
                }
                if (win.empty()) {
                    miss_at = miss.size(); // no room: nothing more this period
                } else {
                    fn_prefetch(win);
                    const int64_t t_end = now + 500000;
                    size_t i = 0;
                    for (; i < win.size() && llama_fn_p5_now_us() < t_end && !t->stop.load(); ++i) {
                        size_t done = 0;
                        const bool all = res_touch({ win[i] }, t->page, act, &done);
                        t->b_rewarm.fetch_add(done);
                        if (!all) {
                            break; // activity: this block again at the next idle sample
                        }
                    }
                    miss_at += i;
                }
            }
        }
        if (now - t_per >= 60000000) {
            t_per = now;
            if (t->resset) {
                res_queue_drop(t, res_droppable(t, true, false)); // pages read back since (a refill, a CPU path)
            }
            if (t->probe_state.load() == 0) {
                // [TAG_FN_L22_SSDFIX] not while a probe reads the set in or watches it; the prompts' low-water first (give-
                // back), and the plan gives marks back only within what that low-water leaves above the cushion
                size_t lw_room = probe.lw_reset ? 0 : SIZE_MAX; // no window with a prompt yet: the plan as before
                if (t->ssd >= 1 && probe.lw_prev != SIZE_MAX && (probe.act_prev || probe.act_cur)) {
                    const size_t lw = std::min(probe.lw_cur, probe.lw_prev);
                    if (lw < GIVE_CUSHION) {
                        res_give_back(t, lw, probe, now);
                        lw_room = 0; // no marks back now; measured again from here
                    } else {
                        lw_room = lw > GIVE_TARGET ? lw - GIVE_TARGET : 0; // marks back keep the give-back's target
                    }
                }
                res_ssd_plan(t, 1, lw_room);
            }
        }
    }
#else
    GGML_UNUSED(t);
#endif
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// the residency interface (llama-fn-l22.h)

uint8_t llama_fn_res_class(const llama_model * model, int il, int e) {
    res_ref r(model);
    if (r.t == nullptr) {
        return LLAMA_FN_RES_RAM;
    }
    const res_layer * L = res_find(r.t, il);
    if (L == nullptr || e < 0 || e >= L->n_exp) {
        return LLAMA_FN_RES_RAM;
    }
    return class_of(L->fl[e].load(std::memory_order_relaxed));
}

bool llama_fn_res_classes(const llama_model * model, int il, uint8_t * out, int n) {
    res_ref r(model);
    if (r.t == nullptr || out == nullptr || n <= 0) {
        return false;
    }
    const res_layer * L = res_find(r.t, il);
    if (L == nullptr) {
        return false;
    }
    for (int e = 0; e < n; ++e) {
        out[e] = e < L->n_exp ? class_of(L->fl[e].load(std::memory_order_relaxed)) : (uint8_t) LLAMA_FN_RES_RAM;
    }
    return true;
}

uint64_t llama_fn_res_gen(const llama_model * model) {
    res_ref r(model);
    return r.t ? r.t->gen.load() : 0;
}

void llama_fn_res_mark_pin(const llama_model * model, int il, int e, bool on) {
    res_ref r(model);
    if (r.t == nullptr) {
        return;
    }
    const res_layer * L = res_find(r.t, il);
    if (L == nullptr || e < 0 || e >= L->n_exp) {
        return;
    }
    const uint8_t f = res_update(L->fl[e], [on](uint8_t x) {
        return on ? (uint8_t) ((x | F_PIN) & ~F_SSD) : (uint8_t) (x & ~F_PIN);
    });
    if (((f & F_PIN) != 0) != on) {
        r.t->gen.fetch_add(1);
    }
}

uint64_t llama_fn_res_stream_bytes(const llama_model * model, uint8_t cls) {
    res_ref r(model);
    if (r.t == nullptr) {
        return 0;
    }
    uint64_t b = 0;
    for (const auto & L : r.t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            b += class_of(L.fl[e].load(std::memory_order_relaxed)) == cls ? L.eb : 0;
        }
    }
    return b;
}

// ---------------------------------------------------------------------------------------------------------------------
// P5 internal (llama-fn-l22-p5.h)

bool llama_fn_p5_ready(const llama_model & model) {
#ifdef _WIN32
    if (model.arch != LLM_ARCH_QWEN4EXP) {
        return false;
    }
    const bool resset = llama_fn_l3_int(model, "LLAMA_FN_L22_RESSET", 0) != 0;
    const bool rewarm = llama_fn_l3_int(model, "LLAMA_FN_L22_REWARM", 0) != 0;
    const int  ssd    = std::max(0, std::min(2, llama_fn_l3_int(model, "LLAMA_FN_L22_SSDTIER", 0)));
    const bool pin    = llama_fn_l3_int(model, "LLAMA_FN_L22_PIN", 0) != 0;
    if (!resset && !rewarm && ssd == 0 && !pin) {
        return false;
    }
    if (g_res.load() != nullptr) {
        return g_res.load()->model == &model;
    }
    auto * t = new res_table();
    t->model  = &model;
    t->resset = resset;
    t->rewarm = rewarm;
    t->ssd    = ssd;
    t->borrow_ram = llama_fn_l22_switch_on(llama_fn_env(model, "LLAMA_FN_L22_UBPLAN")) ||
                    llama_fn_l22_switch_on(llama_fn_env(model, "LLAMA_FN_L22_BIGUB2")); // [TAG_FN_L22_UBPLAN] P10 R2
    t->floor_b = (size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L15_WARM_FREE_MIB", 4096)) << 20;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    t->page = si.dwPageSize ? si.dwPageSize : 4096;
    t->n_layer = (int) std::min<size_t>(model.hparams.n_layer(), model.layers.size());
    t->by_il.assign((size_t) t->n_layer, -1);
    for (int il = 0; il < t->n_layer; ++il) {
        const auto & ML = model.layers[il];
        const ggml_tensor * ts[3] = { ML.ffn_up_exps, ML.ffn_gate_exps, ML.ffn_down_exps };
        res_layer L;
        bool ok = true;
        for (int k = 0; k < 3 && ok; ++k) {
            const ggml_tensor * x = ts[k];
            ok = x && x->buffer && ggml_backend_buffer_is_host(x->buffer) && x->data && x->ne[2] >= 1 &&
                 (k == 0 || x->ne[2] == L.n_exp) && x->nb[2]*x->ne[2] == ggml_nbytes(x);
            if (ok) {
                L.t[k]    = x;
                L.base[k] = (const char *) x->data;
                L.nb[k]   = x->nb[2];
                L.n_exp   = (int) x->ne[2];
            }
        }
        if (!ok) {
            continue;
        }
        L.il = il;
        L.eb = res_slice_bytes(L);
        L.fl.reset(new std::atomic<uint8_t>[(size_t) L.n_exp]());
        L.slot.assign((size_t) L.n_exp, -1);
        L.cnt.assign((size_t) L.n_exp, 0.0f);
        L.vol.assign((size_t) L.n_exp, 0);
        t->by_il[il] = (int) t->L.size();
        t->L.push_back(std::move(L));
    }
    if (t->L.empty()) {
        delete t;
        return false;
    }
    t->lend_known = res_lend_estimate(t);
    // the first view (the hot set's bookkeeping may be busy for a moment)
    for (int li = 0; li < (int) t->L.size(); ++li) {
        for (int k = 0; k < 20; ++k) {
            uint8_t probe = 0;
            int32_t s = -1;
            float   c = 0.0f;
            if (llama_moe_hot_tier_view(t->L[li].t[0], &s, &c, &probe, 0) != 0) {
                break;
            }
            Sleep(10);
        }
        res_poll_layer(t, li, nullptr, nullptr);
    }
    g_res.store(t);
    llama_fn_p5_track(true);
    size_t cls[4] = { 0, 0, 0, 0 };
    for (const auto & L : t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            cls[class_of(L.fl[e].load())] += L.eb;
        }
    }
    LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] residency table: %zu layers, %.2f GiB streamed from the host, %.2f GiB held in VRAM "
            "(resset %d, rewarm %d, ssdtier %d, pin %d)\n", __func__, t->L.size(), cls[0]/1073741824.0, cls[1]/1073741824.0,
            resset ? 1 : 0, rewarm ? 1 : 0, ssd, pin ? 1 : 0);
    // the pin planner starts (its own thread): it marks its experts PIN, the warm pass skips them
    llama_fn_pin_on_ready(&model);
    // R-P6-3: with pins planned (PIN-marked experts, or pins already active) the warm, SSD and re-warm floors stay at least
    // LLAMA_FN_L22_PIN_FLOOR_MIB + 1 GiB of available RAM: the pin valve unpins below its floor and never registers again.
    // [TAG_FN_L23_PINFIX] not any more: pins lock resident pages only and all go under 4 GiB available, so they add no RAM
    // demand; the higher floors put ~10 GiB of the streamed set on the SSD tier (l23p two-connection run). Floors as
    // without pins (LLAMA_FN_L23_PIN_FLOOR_RAISE=1 restores the raise)
    if (pin && llama_fn_l3_int(model, "LLAMA_FN_L23_PIN_FLOOR_RAISE", 0) != 0) {
        bool marked = llama_fn_pin_active();
        for (size_t li = 0; li < t->L.size() && !marked; ++li) {
            for (int e = 0; e < t->L[li].n_exp && !marked; ++e) {
                marked = (t->L[li].fl[e].load() & F_PIN) != 0;
            }
        }
        if (marked) {
            const size_t pf = ((size_t) std::max(0, llama_fn_l3_int(model, "LLAMA_FN_L22_PIN_FLOOR_MIB", 6144)) << 20) + GIB;
            t->floor_b      = std::max(t->floor_b, pf);
            t->rewarm_floor = std::max(t->rewarm_floor, pf);
            LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] pins planned: warm / SSD floor %.1f GiB, re-warm floor %.1f GiB\n", __func__,
                    t->floor_b/1073741824.0, t->rewarm_floor/1073741824.0);
        }
    }
    if (resset) {
        res_ssd_plan(t, 0); // the table-driven warm pass then skips the SSD experts (the old pass reads everything)
    }
    t->svc = std::thread(res_service, t);
    return true;
#else
    GGML_UNUSED(model);
    return false;
#endif
}

bool llama_fn_p5_resset(const llama_model & model) {
    res_ref r(&model);
    return r.t != nullptr && r.t->resset;
}

size_t llama_fn_p5_floor(const llama_model & model) {
    res_ref r(&model);
    return r.t != nullptr ? r.t->floor_b : 0;
}

bool llama_fn_p5_warm_done(const llama_model & model) {
    res_ref r(&model);
    return r.t == nullptr || r.t->warm_done.load();
}

size_t llama_fn_p5_ssd_bytes(const llama_model & model) {
    res_ref r(&model);
    if (r.t == nullptr || r.t->ssd <= 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(r.t->lmtx);
    size_t b = 0;
    for (const auto & L : r.t->L) {
        for (int e = 0; e < L.n_exp; ++e) {
            b += class_of(L.fl[e].load(std::memory_order_relaxed)) == LLAMA_FN_RES_SSD ? L.eb : 0;
        }
    }
    return b;
}

std::vector<llama_fn_p5_piece> llama_fn_p5_warm_list(const llama_model & model, size_t max_bytes, std::vector<size_t> & layer_end,
        size_t * pin_pending, size_t * skipped) {
    std::vector<llama_fn_p5_piece> out;
    *pin_pending = 0;
    *skipped     = 0;
    res_ref r(&model);
    if (r.t == nullptr) {
        return out;
    }
    res_table * t = r.t;
    layer_end.assign((size_t) t->n_layer, 0);
    std::vector<uint8_t> sel;
    std::vector<std::pair<int, int>> runs;
    std::vector<std::pair<int, int>> P;
    for (int il = 0; il < t->n_layer; ++il) {
        const int li = t->by_il[il];
        if (li >= 0) {
            const res_layer & L = t->L[li];
            sel.assign((size_t) L.n_exp, 0);
            for (int e = 0; e < L.n_exp; ++e) {
                const uint8_t f = L.fl[e].load();
                sel[e] = (f & (F_VRAM | F_PIN | F_SSD)) ? 0 : 1;
                if (!sel[e]) {
                    *skipped += L.eb;
                }
                if (f & F_PIN) {
                    P.push_back({ li, e });
                }
            }
            for (int k = 0; k < 3; ++k) {
                runs.clear();
                piece_runs(sel.data(), L.n_exp, L.nb[k], max_bytes, runs);
                for (const auto & r : runs) {
                    out.push_back({ il, k, r.first, r.second, L.base[k] + (size_t) r.first*L.nb[k], (size_t) (r.second - r.first)*L.nb[k] });
                }
            }
        }
        layer_end[il] = out.size();
    }
    std::vector<float> sh;
    res_expert_share(t, P, sh);
    for (size_t i = 0; i < P.size(); ++i) {
        *pin_pending += (size_t) (t->L[P[i].first].eb*(1.0f - sh[i]));
    }
    return out;
}

size_t llama_fn_p5_piece_ranges(const llama_model & model, const llama_fn_p5_piece & pc, std::vector<std::pair<const char *, size_t>> & out) {
    res_ref r(&model);
    if (r.t == nullptr) {
        out.push_back({ pc.p, pc.n });
        return pc.n;
    }
    const res_layer * L = res_find(r.t, pc.il);
    if (L == nullptr) {
        out.push_back({ pc.p, pc.n });
        return pc.n;
    }
    size_t b = 0;
    for (int e = pc.e0; e < pc.e1; ) {
        if (L->fl[e].load() & (F_VRAM | F_PIN | F_SSD)) {
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < pc.e1 && !(L->fl[e1].load() & (F_VRAM | F_PIN | F_SSD))) {
            ++e1;
        }
        const size_t n = (size_t) (e1 - e)*L->nb[pc.part];
        out.push_back({ L->base[pc.part] + (size_t) e*L->nb[pc.part], n });
        b += n;
        e = e1;
    }
    return b;
}

void llama_fn_p5_after_warm(const llama_model & model, size_t warmed, double secs) {
    res_ref r(&model);
    if (r.t == nullptr) {
        return;
    }
    res_table * t = r.t;
    size_t dropped = 0;
    if (t->resset) {
        // [TAG_FN_L22_RESSET] the host copies of VRAM-held experts (and of resident SSD-marked ones) to the standby list
        const auto d = res_droppable(t, true, t->ssd > 0);
        for (const auto & x : d) {
            dropped += t->L[x.first].eb;
        }
        if (res_quiet()) {
            res_drop(t, d); // now: the stage-1 plan below sees the room it frees
        } else {
            res_queue_drop(t, d);
        }
    }
    res_ssd_plan(t, 1);
    double share = 1.0;
    std::vector<range> miss;
    res_missing_blocks(t, miss, &share);
    t->share.store(share);
    t->warm_done.store(true);
    if (t->resset) {
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] ready: %.2f GiB warmed in %.1f s, %.2f GiB of VRAM-held host copies go to standby, "
                "streamed set %.1f %% in RAM\n", __func__, warmed/1073741824.0, secs, dropped/1073741824.0, 100.0*share);
    } else {
        LLAMA_LOG_INFO("%s: [TAG_FN_L22_RESSET] ready after the full warm pass (resset off): streamed set %.1f %% in RAM\n",
                __func__, 100.0*share);
    }
    res_log(t, "ready");
}

void llama_fn_p5_free(const llama_model & model) {
    res_table * t = g_res.load();
    if (t != nullptr && t->model == &model) {
        t->stop.store(true);
        if (t->svc.joinable()) {
            t->svc.join();
        }
        llama_fn_p5_track(false);
        g_res.store(nullptr);
        while (g_res_readers.load() != 0) {
            std::this_thread::yield();
        }
        delete t;
    }
    llama_fn_p5_dio_free(model);
}
