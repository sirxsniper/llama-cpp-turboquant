// [TAG_FN_L4_HOST] LLAMA_FN_L4_HOSTPROF=1|N (qwen4exp): exclusive host time per named segment of the decode loop.
// The thread that makes the first call after the switch owns the profile; calls from other threads do nothing.
// Each llama_hp_switch charges the time since the last switch to the segment that was current. A step runs from one
// target graph launch to the next; steps around a wide (prompt) launch and steps longer than 250 ms (idle) are dropped.

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-fn-auto.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

struct hp_state {
    bool     on    = false;
    int64_t  every = 256;
    std::thread::id owner;
    bool     owned = false;

    int      cur    = LLAMA_HP_OTHER;
    int64_t  t_last = 0;

    // the step in progress
    int64_t  step_t0 = 0;
    bool     step_ok = false;
    int64_t  s_ns[LLAMA_HP_N_SEG] = {};
    int64_t  s_n [LLAMA_HP_N_SEG] = {};

    // the valid steps since the last report
    int64_t  n_steps = 0;
    int64_t  n_drop  = 0;
    int64_t  tot_step_ns = 0;
    int64_t  tot_ns[LLAMA_HP_N_SEG] = {};
    int64_t  tot_n [LLAMA_HP_N_SEG] = {};

    // the class of each context that decoded (0 target, 1 MTP process, 2 MTP draft)
    const void * ctx_key[8] = {};
    int          ctx_cls[8] = {};
};

hp_state g_hp;

int64_t hp_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool hp_mine() {
    if (!g_hp.on) {
        return false;
    }
    if (!g_hp.owned) {
        g_hp.owner = std::this_thread::get_id();
        g_hp.owned = true;
        g_hp.t_last = hp_now_ns();
        return true;
    }
    return g_hp.owner == std::this_thread::get_id();
}

const char * hp_dec_name(int sub) {
    static const char * names[LLAMA_HP_DEC_N] = { "prep", "apply", "graph", "inputs", "launch", "wait", "bend", "out", "post", "sync",
                                                  "reset", "alloc" };
    return sub >= 0 && sub < LLAMA_HP_DEC_N ? names[sub] : "?";
}

std::string hp_seg_name(int seg) {
    switch (seg) {
        case LLAMA_HP_OTHER:      return "other";
        case LLAMA_HP_SRV_PRE:    return "srv.pre";
        case LLAMA_HP_SRV_CKPT:   return "srv.ckpt";
        case LLAMA_HP_SRV_BATCH:  return "srv.batch";
        case LLAMA_HP_SRV_DPOST:  return "srv.dpost";
        case LLAMA_HP_SRV_SAMPLE: return "srv.sample";
        case LLAMA_HP_SRV_BOOK:   return "srv.book";
        case LLAMA_HP_SRV_TOKEN:  return "srv.token";
        case LLAMA_HP_MTP_PROC:   return "mtp.proc";
        case LLAMA_HP_MTP_DRAFT:  return "mtp.draft";
        case LLAMA_HP_MTP_ACCEPT: return "mtp.accept";
        default: break;
    }
    if (seg >= LLAMA_HP_DEC_BASE && seg < LLAMA_HP_N_SEG) {
        static const char cls[3] = { 'T', 'P', 'D' };
        const int k = seg - LLAMA_HP_DEC_BASE;
        return std::string(1, cls[k / LLAMA_HP_DEC_N]) + "." + hp_dec_name(k % LLAMA_HP_DEC_N);
    }
    return "?";
}

void hp_report() {
    const double n = (double) g_hp.n_steps;
    std::vector<int> order;
    for (int s = 0; s < LLAMA_HP_N_SEG; ++s) {
        if (g_hp.tot_n[s] > 0) {
            order.push_back(s);
        }
    }
    std::sort(order.begin(), order.end(), [](int a, int b) { return g_hp.tot_ns[a] > g_hp.tot_ns[b]; });
    std::string line;
    char buf[96];
    for (int s : order) {
        snprintf(buf, sizeof(buf), " %s %.1f/%.2f", hp_seg_name(s).c_str(), g_hp.tot_ns[s]/1e3/n, g_hp.tot_n[s]/n);
        line += buf;
    }
    LLAMA_LOG_INFO("hostprof: [TAG_FN_L4_HOST] %" PRId64 " steps (%" PRId64 " dropped), %.1f us per step; per step us/calls:%s\n",
            g_hp.n_steps, g_hp.n_drop, g_hp.tot_step_ns/1e3/n, line.c_str());
    g_hp.n_steps = 0;
    g_hp.n_drop  = 0;
    g_hp.tot_step_ns = 0;
    std::fill(std::begin(g_hp.tot_ns), std::end(g_hp.tot_ns), 0);
    std::fill(std::begin(g_hp.tot_n),  std::end(g_hp.tot_n),  0);
}

} // namespace

bool llama_fn_l4_host_flag(const llama_model * model, const char * name) {
    if (model == nullptr) {
        return false;
    }
    const int v = llama_fn_l3_int(*model, name, -1);
    if (v >= 0) {
        return v != 0;
    }
    return llama_fn_l3_int(*model, "LLAMA_FN_L4_HOST", 0) != 0;
}

void llama_hp_enable(int every) {
    if (g_hp.on) {
        return;
    }
    g_hp.every = every > 1 ? every : 256;
    g_hp.on    = true;
    LLAMA_LOG_INFO("%s: [TAG_FN_L4_HOST] host profile of the decode loop, a report every %" PRId64 " steps\n", __func__, g_hp.every);
}

bool llama_hp_on(void) {
    return g_hp.on;
}

int llama_hp_switch(int seg) {
    if (!hp_mine()) {
        return -1;
    }
    const int64_t t = hp_now_ns();
    const int prev = g_hp.cur;
    g_hp.s_ns[prev] += t - g_hp.t_last;
    g_hp.s_n[prev]  += 1;
    g_hp.t_last = t;
    g_hp.cur    = seg >= 0 && seg < LLAMA_HP_N_SEG ? seg : LLAMA_HP_OTHER;
    return prev;
}

int llama_hp_current(void) {
    return hp_mine() ? g_hp.cur : -1;
}

void llama_hp_ctx_class_set(const void * ctx, int cls) {
    if (!hp_mine()) {
        return;
    }
    for (int i = 0; i < 8; ++i) {
        if (g_hp.ctx_key[i] == ctx || g_hp.ctx_key[i] == nullptr) {
            g_hp.ctx_key[i] = ctx;
            g_hp.ctx_cls[i] = cls;
            return;
        }
    }
}

int llama_hp_ctx_class_get(const void * ctx) {
    for (int i = 0; i < 8; ++i) {
        if (g_hp.ctx_key[i] == ctx) {
            return g_hp.ctx_cls[i];
        }
    }
    return 0;
}

int llama_hp_class_of_caller(void) {
    const int c = llama_hp_current();
    if (c == LLAMA_HP_MTP_PROC || (c >= LLAMA_HP_DEC_BASE + LLAMA_HP_DEC_N && c < LLAMA_HP_DEC_BASE + 2*LLAMA_HP_DEC_N)) {
        return 1;
    }
    if (c == LLAMA_HP_MTP_DRAFT || c >= LLAMA_HP_DEC_BASE + 2*LLAMA_HP_DEC_N) {
        return 2;
    }
    return 0;
}

void llama_hp_step(int n_tokens) {
    if (!hp_mine()) {
        return;
    }
    llama_hp_switch(g_hp.cur); // close the open segment at this instant
    const int64_t t = g_hp.t_last;
    const bool ok_now = n_tokens <= 16;
    if (g_hp.step_t0 > 0) {
        const int64_t dt = t - g_hp.step_t0;
        if (g_hp.step_ok && ok_now && dt < 250*1000*1000LL) {
            g_hp.n_steps++;
            g_hp.tot_step_ns += dt;
            for (int s = 0; s < LLAMA_HP_N_SEG; ++s) {
                g_hp.tot_ns[s] += g_hp.s_ns[s];
                g_hp.tot_n[s]  += g_hp.s_n[s];
            }
        } else {
            g_hp.n_drop++;
        }
    }
    std::fill(std::begin(g_hp.s_ns), std::end(g_hp.s_ns), 0);
    std::fill(std::begin(g_hp.s_n),  std::end(g_hp.s_n),  0);
    if (g_hp.n_steps >= g_hp.every) {
        hp_report();
    }
    // the report's own time belongs to no step
    g_hp.t_last  = hp_now_ns();
    g_hp.step_t0 = g_hp.t_last;
    g_hp.step_ok = ok_now;
}
