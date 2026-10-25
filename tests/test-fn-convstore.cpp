// [TAG_FN_L19_CONVSTORE] the policy of the agent conversation store (tools/server/server-convstore.h, header only): which
// checkpoint a full ring drops, the eviction value of a stored conversation, the thinning of checkpoints on disk. CPU only,
// no model.
//
// Checks: fixed cases; an agent-loop replay (a system + tools block, then turns that each add a checkpoint at the user
// message and one at the assistant-turn start) in which the old rule (min-step spacing, then the oldest) loses the
// system-block checkpoint and the new one keeps it, the last user message and the newest; random rings (the victim is
// never the incoming checkpoint, and never a protected one while an unprotected one exists); the greedy-dual value
// (monotone in tokens, restores and bytes; aging).

#include "../tools/server/server-convstore.h"

#include <algorithm>
#include <cstdio>
#include <random>
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

static void test_victim_fixed() {
    // ring 8000 (system block), 17000, 26000, 35000; incoming 36000; last user message at 35500
    {
        const std::vector<int64_t> ns = { 8000, 17000, 26000, 35000, 36000 };
        const int v = convstore_ckpt_victim(ns, 8000, 35500);
        TCHECK(v == 2, "victim %d, want 2 (26000: not the system block, not the last user message, not halfway)", v);
    }
    // the system block end unknown: the oldest is protected
    {
        const std::vector<int64_t> ns = { 100, 5000, 5100, 20000, 20004 };
        const int v = convstore_ckpt_victim(ns, -1, -1);
        // protected: 100 (oldest), 20004 (newest), halfway 10002 -> 5100; left 5000 and 20000: gaps 4900 and 14900
        TCHECK(v == 1, "victim %d, want 1", v);
    }
    // smallest gap among the unprotected, the older of equals
    {
        const std::vector<int64_t> ns = { 1000, 2000, 3000, 4000, 9000, 9004 };
        const int v = convstore_ckpt_victim(ns, 1000, 9000);
        // protected: 1000, 9000 (last user), halfway 4502 -> 4000, 9004; left 2000 (gap 1000), 3000 (gap 1000) -> 2000
        TCHECK(v == 1, "victim %d, want 1", v);
    }
    // a ring of one: its only entry goes
    {
        const std::vector<int64_t> ns = { 500, 900 };
        TCHECK(convstore_ckpt_victim(ns, 500, 500) == 0, "ring of one");
    }
    TCHECK(convstore_ckpt_victim({ 7 }, 0, 0) == -1, "nothing to drop");
    TCHECK(convstore_ckpt_victim({}, 0, 0) == -1, "empty");
}

// the old rule of create_checkpoint() for a full ring ([TAG_SYNC_CKPT_EVICT]): entries within min_step of an earlier one
// (not of the current task) go, then the oldest
static void old_rule(std::vector<int64_t> & ring, std::vector<int> & task, int cur_task, size_t n_max, int64_t min_step) {
    if (ring.size() + 1 >= n_max) {
        int64_t last = -1;
        for (size_t i = 0; i < ring.size();) {
            if (task[i] != cur_task && last >= 0 && ring[i] <= last + min_step) {
                ring.erase(ring.begin() + i);
                task.erase(task.begin() + i);
                continue;
            }
            last = ring[i];
            ++i;
        }
    }
    while (ring.size() >= n_max) {
        ring.erase(ring.begin());
        task.erase(task.begin());
    }
}

static void test_agent_loop() {
    const size_t  n_max    = 4;      // Flash-Next: 512 MiB / 112.6 MiB
    const int64_t min_step = 8192;
    const int64_t sys_end  = 9000;   // system + tools block

    std::vector<int64_t> ring_old;
    std::vector<int>     task_old;
    std::vector<int64_t> ring_new;

    std::mt19937 rng(42);
    int64_t n = sys_end;
    bool old_lost_sys = false;
    for (int turn = 0; turn < 60; ++turn) {
        // the turn: a user message (or a tool result) at n, then its tokens, then the assistant-turn start at the end
        const int64_t user = n;
        const int64_t len  = 300 + (int64_t) (rng() % 4000);
        const int64_t tail = user + len - 4;
        for (int64_t pos : { user, tail }) {
            if (ring_old.empty() || pos == user || pos > ring_old.back() + min_step || pos == tail) {
                old_rule(ring_old, task_old, turn, n_max, min_step);
                ring_old.push_back(pos);
                task_old.push_back(turn);
            }
            if (ring_new.size() >= n_max) {
                std::vector<int64_t> ns = ring_new;
                ns.push_back(pos);
                const int v = convstore_ckpt_victim(ns, sys_end, user);
                TCHECK(v >= 0 && v < (int) ring_new.size(), "turn %d: victim %d out of range", turn, v);
                ring_new.erase(ring_new.begin() + v);
            }
            ring_new.push_back(pos);
            TCHECK(std::is_sorted(ring_new.begin(), ring_new.end()), "turn %d: ring not ascending", turn);
            TCHECK(ring_new.size() <= n_max, "turn %d: ring of %zu", turn, ring_new.size());
        }
        // the first turn's user checkpoint is the end of the system block
        const bool new_has_sys = std::find(ring_new.begin(), ring_new.end(), sys_end) != ring_new.end();
        const bool new_has_user = std::find(ring_new.begin(), ring_new.end(), user) != ring_new.end();
        const bool new_has_tail = ring_new.back() == tail;
        TCHECK(new_has_sys, "turn %d: the new rule lost the system-block checkpoint", turn);
        TCHECK(new_has_user, "turn %d: the new rule lost the last user message", turn);
        TCHECK(new_has_tail, "turn %d: the new rule lost the newest", turn);
        // one more point, between about a quarter of the way and the end (a ladder with one rung)
        if (tail > 4 * sys_end) {
            TCHECK(ring_new.size() == n_max && ring_new[1] * 9 >= tail * 2 && ring_new[1] < user,
                   "turn %d: no point between a quarter of the way and the end (n = %lld, ring[1] = %lld)", turn,
                   (long long) tail, (long long) ring_new[1]);
        }
        old_lost_sys = old_lost_sys || std::find(ring_old.begin(), ring_old.end(), sys_end) == ring_old.end();
        n = user + len;
    }
    TCHECK(old_lost_sys, "the old rule was expected to lose the system-block checkpoint in a long loop");
    printf("agent loop: 60 turns to %lld tokens, the old rule %s the system-block checkpoint, the new keeps it\n",
           (long long) n, old_lost_sys ? "lost" : "kept");
}

static void test_victim_random() {
    std::mt19937 rng(7);
    for (int it = 0; it < 20000; ++it) {
        const int n = 2 + (int) (rng() % 8);
        std::vector<int64_t> ns;
        int64_t p = (int64_t) (rng() % 100);
        for (int i = 0; i < n; ++i) {
            ns.push_back(p);
            p += 1 + (int64_t) (rng() % 5000);
        }
        const int64_t sys_end   = (rng() % 3 == 0) ? -1 : ns[rng() % n];
        const int64_t last_user = (rng() % 3 == 0) ? -1 : ns[n - 1] - (int64_t) (rng() % 3000);
        const int v = convstore_ckpt_victim(ns, sys_end, last_user);
        TCHECK(v >= 0 && v < n - 1, "random %d: victim %d of %d", it, v, n);
        if (n >= 5) {
            // the system-block one (the newest at or before sys_end, or the oldest) survives a ring of 4 or more
            int s = 0;
            if (sys_end >= 0) {
                s = -1;
                for (int i = 0; i < n - 1; ++i) {
                    if (ns[i] <= sys_end) {
                        s = i;
                    }
                }
                if (s < 0) {
                    s = 0;
                }
            }
            TCHECK(v != s, "random %d: dropped the system-block checkpoint %d", it, s);
        }
    }
}

static void test_gdsf() {
    const uint64_t GiB = 1024ull * 1024 * 1024;
    const double a = convstore_gdsf(0.0, 0, 30000, GiB, 2048);
    const double b = convstore_gdsf(0.0, 1, 30000, GiB, 2048);
    const double c = convstore_gdsf(0.0, 0, 30000, 2 * GiB, 2048);
    const double d = convstore_gdsf(0.0, 0, 60000, GiB, 2048);
    const double e = convstore_gdsf(1e6, 0, 30000, GiB, 2048);
    TCHECK(b > a, "a restore raises the value");
    TCHECK(c < a, "more bytes lower it");
    TCHECK(d > a, "more tokens raise it");
    TCHECK(e > a + 9e5, "aging adds L");
    TCHECK(convstore_gdsf(0.0, 0, 0, 0, 0) == 0.0, "zero");
    // below the share of one entry, bytes do not count
    TCHECK(convstore_gdsf(0.0, 0, 2000, GiB / 8, 2048, 2 * GiB) < convstore_gdsf(0.0, 0, 30000, GiB, 2048, 2 * GiB),
           "a small side request is worth less than a session while the count binds");
    TCHECK(convstore_gdsf(0.0, 0, 2000, GiB / 8, 2048) > convstore_gdsf(0.0, 0, 30000, GiB, 2048),
           "without the share, per byte the small one is worth more");

    // replay: 3 sessions of ~1 GiB take turns with title requests of 2K tokens / 0.13 GiB; 6 entries of room (a 32 GiB
    // budget over 16 entries: 2 GiB each). The sessions must outlive the one-off title requests.
    struct ent { int id; double v; int64_t hits; int64_t n; uint64_t bytes; };
    std::vector<ent> store;
    double L = 0;
    auto put = [&](int id, int64_t hits, int64_t n, uint64_t bytes) {
        while (store.size() >= 6) {
            auto it = std::min_element(store.begin(), store.end(), [](const ent & x, const ent & y) { return x.v < y.v; });
            L = std::max(L, it->v);
            store.erase(it);
        }
        store.push_back({ id, convstore_gdsf(L, hits, n, bytes, 2048, 32 * GiB / 16), hits, n, bytes });
    };
    for (int round = 0; round < 20; ++round) {
        for (int s = 0; s < 3; ++s) {
            put(s, round, 30000 + 2000 * round, GiB + (uint64_t) round * 64 * 1024 * 1024);
            store.erase(std::remove_if(store.begin(), store.end(), [&](const ent & x) { return x.id == s && x.hits < round; }), store.end());
        }
        for (int t = 0; t < 4; ++t) {
            put(100 + round * 10 + t, 0, 2000, GiB / 8);
        }
        int alive = 0;
        for (const auto & x : store) {
            alive += x.id < 3;
        }
        TCHECK(alive == 3, "round %d: %d of the 3 sessions stored", round, alive);
    }
}

static void test_thin() {
    TCHECK(convstore_thin({ 1, 2, 3 }, 3) == -1, "within the limit");
    TCHECK(convstore_thin({ 1000, 1100, 5000, 9000 }, 3) == 1, "smallest gap");
    TCHECK(convstore_thin({ 1000, 2000, 3000, 3001 }, 3) == 0 || convstore_thin({ 1000, 2000, 3000, 3001 }, 3) == 1,
           "never the newest");
    TCHECK(convstore_thin({ 5, 6 }, 1) == 0, "the older of two");
    TCHECK(convstore_near({ 100, 5000 }, 900, 1024), "near");
    TCHECK(!convstore_near({ 100, 5000 }, 2000, 1024), "not near");
    TCHECK(!convstore_near({}, 2000, 1024), "nothing kept");
}

// [TAG_FN_L20_CONVFIX_LAZY] the ring and the checkpoints on disk together keep the newest cap positions
static void test_keep_from() {
    const int64_t lo = std::numeric_limits<int64_t>::min();
    TCHECK(convfix_keep_from({}, 4) == lo, "empty");
    TCHECK(convfix_keep_from({5657, 8025, 8021}, 4) == lo, "3 fit in 4");
    TCHECK(convfix_keep_from({5657, 8021, 8025, 10715}, 4) == lo, "4 fit in 4");
    // after a restore: the ring holds the new prompt's 2, the disk the stored ring's 4
    TCHECK(convfix_keep_from({10719, 15570, 5657, 8021, 8025, 10715}, 4) == 8025, "newest 4 of 6");
    TCHECK(convfix_keep_from({1, 2, 3}, 1) == 3, "cap 1");
    TCHECK(convfix_keep_from({7, 7, 3}, 2) == 7, "duplicates count once each");
    TCHECK(convfix_keep_from({1, 2}, 0) == lo, "cap 0 keeps all");
    // random: exactly min(cap, n) positions are >= the result, and the result is one of them
    std::mt19937 rng(20);
    for (int it = 0; it < 2000; ++it) {
        const size_t n   = rng() % 12;
        const size_t cap = 1 + rng() % 6;
        std::vector<int64_t> pos;
        for (size_t i = 0; i < n; ++i) {
            pos.push_back((int64_t) (rng() % 50000));
        }
        const int64_t k = convfix_keep_from(pos, cap);
        size_t n_ge = 0;
        for (int64_t p : pos) {
            n_ge += p >= k;
        }
        if (n <= cap) {
            TCHECK(k == lo && n_ge == n, "all fit (n %zu cap %zu)", n, cap);
        } else {
            TCHECK(std::find(pos.begin(), pos.end(), k) != pos.end(), "result is a position");
            size_t n_gt = 0;
            for (int64_t p : pos) {
                n_gt += p > k;
            }
            TCHECK(n_gt < cap && n_ge >= cap, "cap-th newest (n %zu cap %zu: > %zu, >= %zu)", n, cap, n_gt, n_ge);
        }
    }
}

// [TAG_FN_L24_KEEPVAL] the store by value: the owner's 10-22 sequence (llama-1792637140:12212..12626), n_max 4
static void test_keepval() {
    struct ent { int64_t n; int64_t hits; double t_use; };
    const double fixed = 2048.0, half = 900.0, active = 3600.0;
    auto view = [&](const std::vector<ent> & st, double now, int keep) {
        std::vector<convstore_val_entry> es;
        for (int i = 0; i < (int) st.size(); ++i) {
            es.push_back({ st[i].n, st[i].hits, now - st[i].t_use, i == keep });
        }
        return es;
    };
    // store one entry at time now: evict for the count first (as server_keepctx::save with val on)
    auto store = [&](std::vector<ent> & st, ent e, double now, size_t n_max) {
        while (st.size() >= n_max) {
            const int v = convstore_val_victim(view(st, now, -1), fixed, half, active, false);
            if (v < 0) {
                break;
            }
            st.erase(st.begin() + v);
        }
        st.push_back(e);
    };
    auto has = [](const std::vector<ent> & st, int64_t n) {
        for (const auto & e : st) {
            if (e.n == n) {
                return true;
            }
        }
        return false;
    };
    {
        std::vector<ent> st = { { 129130, 1, -600.0 }, { 15846, 0, -500.0 }, { 4329, 1, -400.0 } };
        store(st, { 48656, 0, 0.0 }, 0.0, 4);       // the main design conversation
        st[std::min<size_t>(2, st.size() - 1)].t_use = 1.0; // the 4329-token entry restored (kept) right after
        store(st, { 17905, 0, 115.0 }, 115.0, 4);   // side conversations
        store(st, { 16670, 0, 200.0 }, 200.0, 4);
        store(st, { 17469, 0, 323.0 }, 323.0, 4);
        store(st, { 5200, 0, 334.0 }, 334.0, 4);
        TCHECK(has(st, 48656), "the main conversation survives 4 side stores");
        TCHECK(st.size() == 4, "count %zu", st.size());
    }
    {
        // the rule of before (least recently stored first, LLAMA_FN_L19_CONVSTORE_EVICT 0) lost it on the 4th side store
        std::vector<int64_t> st = { 48656 };
        for (int64_t n : { 17905, 16670, 17469, 5200 }) {
            while (st.size() >= 4) {
                st.erase(st.begin());
            }
            st.push_back(n);
        }
        TCHECK(std::find(st.begin(), st.end(), 48656) == st.end(), "the old rule was expected to evict the main conversation");
    }
    {
        // 10 one-shot side requests in a row: the main conversation stays (protected while active)
        std::vector<ent> st = { { 48656, 0, 0.0 } };
        for (int i = 0; i < 10; ++i) {
            store(st, { 20000 + i, 0, 30.0 * (i + 1) }, 30.0 * (i + 1), 4);
        }
        TCHECK(has(st, 48656), "the main conversation survives 10 bigger-than-usual side stores");
        // the newest side requests stay, the oldest went first among equals by age
        TCHECK(has(st, 20009) && !has(st, 20000), "lowest value = oldest side request");
    }
    {
        // not used for 2 h: no longer protected, a recent bigger conversation is worth more
        std::vector<convstore_val_entry> es = { { 48656, 0, 7200.0, false }, { 60000, 0, 10.0, false }, { 52000, 0, 20.0, false } };
        TCHECK(convstore_val_protect(es, active) == 1, "the biggest active one");
        TCHECK(convstore_val_victim(es, fixed, half, active, false) == 0, "the stale big conversation goes for the count");
    }
    {
        // keep is never evicted; the count never takes the protected one; the budget does, last
        std::vector<convstore_val_entry> es = { { 48656, 0, 60.0, false }, { 3000, 0, 5.0, true } };
        TCHECK(convstore_val_victim(es, fixed, half, active, false) == -1, "count: only keep and protected");
        TCHECK(convstore_val_victim(es, fixed, half, active, true) == 0, "budget: the protected one last");
        es[0].keep = true;
        TCHECK(convstore_val_victim(es, fixed, half, active, true) == -1, "budget: never keep");
        std::vector<convstore_val_entry> es2 = { { 48656, 0, 60.0, false }, { 3000, 0, 5.0, false }, { 9000, 0, 5.0, false } };
        TCHECK(convstore_val_victim(es2, fixed, half, active, true) == 1, "budget: the lowest value first");
    }
    {
        // recency and restores
        TCHECK(convstore_val(10000, 0, 900.0, 0.0, 900.0) == 5000.0, "half at half_s");
        TCHECK(convstore_val(10000, 1, 0.0, 0.0, 900.0) == 20000.0, "one restore doubles");
        TCHECK(convstore_val(10000, 99, 0.0, 0.0, 900.0) == 40000.0, "restores capped at 3");
        TCHECK(convstore_val(-5, -1, -3.0, 2048.0, 900.0) == 2048.0, "negative inputs clamp");
        TCHECK(convstore_val_protect({}, active) == -1 && convstore_val_victim({}, fixed, half, active, true) == -1, "empty");
    }
}

int main() {
    test_victim_fixed();
    test_agent_loop();
    test_victim_random();
    test_gdsf();
    test_thin();
    test_keep_from();
    test_keepval();
    printf("test-fn-convstore: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
