// [TAG_FN_L22] round 22, P0's own items (src/llama-fn-l22.h): switch parsing, the decode-phase rule (fn_gen) and the X
// lever bits, the expert host ranges, the PASSLOG helper ([TAG_FN_L22_PASSLOG]) and llama_fn_pin_on_oom_ext.

#include "llama-fn-l22.h"

#include "llama-ext.h"
#include "llama-fn-auto.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-moe-bridge.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#else
#    if !defined(__wasi__) // WASI has no getrusage
#        include <sys/resource.h>
#    endif
#    include <unistd.h>
#endif

//
// switches
//

bool llama_fn_l22_switch_on(const char * v) {
    if (v == nullptr) {
        return false;
    }
    std::string s;
    for (const char * c = v; *c; ++c) {
        if (!isspace((unsigned char) *c)) {
            s.push_back((char) tolower((unsigned char) *c));
        }
    }
    if (s.empty() || s == "off" || s == "false" || s == "no") {
        return false;
    }
    char * end = nullptr;
    const double d = strtod(s.c_str(), &end);
    if (end != nullptr && end != s.c_str() && *end == '\0') {
        return d != 0.0;
    }
    return true; // a word (e.g. "check", "on", "raw")
}

static const char * fn_l22_env(const llama_model * model, const char * name) {
    return model ? llama_fn_env(*model, name) : getenv(name);
}

llama_fn_l22_x_cfg llama_fn_l22_x_read(const llama_model * model) {
    llama_fn_l22_x_cfg cfg;
    if (const char * v = fn_l22_env(model, "LLAMA_FN_L22_X_TAILSKIP")) {
        cfg.tailskip = std::max(0, atoi(v));
    }
    cfg.tail_multi = llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_TAILSKIP_MULTI"));
    cfg.prior      = llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_PRIOR"));
    cfg.defer      = llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_DEFER"));
    cfg.resident   = llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_RESIDENT"));
    return cfg;
}

uint8_t llama_fn_l22_hooks_read(const llama_model * model) {
    uint8_t h = 0;
    h |= llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_CPUSHARE")) ? LLAMA_FN_L22_HOOK_CPUSHARE : 0;
    h |= llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_SAMPSTATIC")) ? LLAMA_FN_L22_HOOK_SAMPSTATIC : 0;
    h |= llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_PRIOR")) ||
         llama_fn_l22_switch_on(fn_l22_env(model, "LLAMA_FN_L22_X_RESIDENT")) ? LLAMA_FN_L22_HOOK_PRIOR : 0;
    return h;
}

uint8_t llama_fn_l22_gen_rule(int32_t phase, uint32_t n_tokens, uint32_t n_outputs, bool x_any) {
    return phase == LLAMA_FN_PHASE_GEN && x_any && n_tokens > 0 && n_outputs == n_tokens ? 1 : 0;
}

uint8_t llama_fn_l22_x_bits(const llama_fn_l22_x_cfg & cfg, uint8_t fn_gen, uint32_t n_seqs_unq) {
    if (!fn_gen) {
        return 0;
    }
    uint8_t b = 0;
    if (cfg.tailskip > 0 && (n_seqs_unq == 1 || cfg.tail_multi)) {
        b |= LLAMA_FN_L22_XB_TAIL;
    }
    b |= cfg.prior    ? LLAMA_FN_L22_XB_PRIOR    : 0;
    b |= cfg.defer    ? LLAMA_FN_L22_XB_DEFER    : 0;
    b |= cfg.resident ? LLAMA_FN_L22_XB_RESIDENT : 0;
    return b;
}

// R-REV-6: plan 3.8, the levers act only through fn_x (0 unless fn_gen)
int32_t llama_fn_l22_job_flags(uint8_t fn_x) {
    return (fn_x & LLAMA_FN_L22_XB_TAIL) ? GGML_MOE_BRIDGE_JOB_TAIL : 0;
}

bool llama_fn_l22_defer_acts(uint8_t fn_x, uint32_t n_seqs_unq, bool have_bridge, bool tbo) {
    return (fn_x & LLAMA_FN_L22_XB_DEFER) != 0 && n_seqs_unq == 1 && have_bridge && !tbo;
}

bool llama_fn_l22_prior_acts(uint8_t fn_x, uint8_t fn_gen, bool prompt_rule) {
    return (fn_x & (LLAMA_FN_L22_XB_PRIOR | LLAMA_FN_L22_XB_RESIDENT)) != 0 || (prompt_rule && !fn_gen);
}

//
// expert ranges
//

bool llama_fn_expert_host_range(const llama_model * model, int il, int e, int part, const uint8_t ** p, size_t * n) {
    if (p) {
        *p = nullptr;
    }
    if (n) {
        *n = 0;
    }
    if (model == nullptr || il < 0 || il >= (int) model->layers.size() || e < 0 || part < 0 || part > 2) {
        return false;
    }
    const auto & layer = model->layers[il];
    const ggml_tensor * t = part == 0 ? layer.ffn_up_exps : part == 1 ? layer.ffn_gate_exps : layer.ffn_down_exps;
    if (t == nullptr || t->data == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer) ||
        e >= t->ne[2]) {
        return false;
    }
    if (p) {
        *p = (const uint8_t *) t->data + (size_t) e*t->nb[2];
    }
    if (n) {
        *n = t->nb[2];
    }
    return true;
}

//
// PASSLOG
//

static std::atomic<int> g_passlog_profile{-1}; // the profile's value of a loaded model, -1 none

static int fn_l22_level_of(const char * v) {
    return v && *v ? std::max(0, std::min(2, atoi(v))) : -1;
}

int llama_fn_l22_passlog_level() {
    static const int env = fn_l22_level_of(getenv("LLAMA_FN_L22_PASSLOG"));
    if (env >= 0) {
        return env;
    }
    const int p = g_passlog_profile.load(std::memory_order_relaxed);
    return p >= 0 ? p : 0;
}

// R-REV-11: ggml-cuda's kind=graph lines follow the resolved level (ggml cannot read the profile): one call per change
static void fn_l22_passlog_push() {
    using set_fn_t = void (*)(int);
    const int lvl = llama_fn_l22_passlog_level();
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        auto fn = reg ? (set_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_passlog_level") : nullptr;
        if (fn) {
            fn(lvl);
        }
    }
}

void llama_fn_l22_passlog_model(const llama_model * model, bool loaded) {
    if (model == nullptr) {
        return;
    }
    if (!loaded) {
        g_passlog_profile.store(-1, std::memory_order_relaxed);
        fn_l22_passlog_push();
        return;
    }
    const int v = fn_l22_level_of(llama_fn_env(*model, "LLAMA_FN_L22_PASSLOG"));
    if (v >= 0) {
        g_passlog_profile.store(v, std::memory_order_relaxed);
    }
    fn_l22_passlog_push();
}

#if defined(_WIN32)
namespace {

// SYSTEM_PERFORMANCE_INFORMATION (ntexapi.h, x64), up to the page read counts
struct fn_l22_sys_perf {
    LARGE_INTEGER IdleProcessTime;
    LARGE_INTEGER IoReadTransferCount;
    LARGE_INTEGER IoWriteTransferCount;
    LARGE_INTEGER IoOtherTransferCount;
    ULONG IoReadOperationCount;
    ULONG IoWriteOperationCount;
    ULONG IoOtherOperationCount;
    ULONG AvailablePages;
    ULONG CommittedPages;
    ULONG CommitLimit;
    ULONG PeakCommitment;
    ULONG PageFaultCount;
    ULONG CopyOnWriteCount;
    ULONG TransitionCount;
    ULONG CacheTransitionCount;
    ULONG DemandZeroCount;
    ULONG PageReadCount;
    ULONG PageReadIoCount;
};

// SYSTEM_MEMORY_LIST_INFORMATION (ntexapi.h)
struct fn_l22_mem_list {
    ULONG_PTR ZeroPageCount;
    ULONG_PTR FreePageCount;
    ULONG_PTR ModifiedPageCount;
    ULONG_PTR ModifiedNoWritePageCount;
    ULONG_PTR BadPageCount;
    ULONG_PTR PageCountByPriority[8];
    ULONG_PTR RepurposedPagesByPriority[8];
    ULONG_PTR ModifiedPageCountPageFile;
};

using fn_l22_nt_query_t = LONG (WINAPI *)(ULONG, PVOID, ULONG, PULONG);

fn_l22_nt_query_t fn_l22_nt_query() {
    static const fn_l22_nt_query_t q = [] {
        HMODULE nt = GetModuleHandleW(L"ntdll.dll");
        return nt ? (fn_l22_nt_query_t) (void *) GetProcAddress(nt, "NtQuerySystemInformation") : nullptr;
    }();
    return q;
}

std::atomic<bool> g_standby_denied{false}; // SystemMemoryListInformation needs a privilege: asked once, then 0

} // namespace
#endif

llama_fn_l22_counters llama_fn_l22_counters_now() {
    llama_fn_l22_counters c = {0, 0, 0, 0};
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        c.faults = pmc.PageFaultCount;
    }
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        c.avail_ram = ms.ullAvailPhys;
    }
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const uint64_t page = si.dwPageSize ? si.dwPageSize : 4096;
    if (fn_l22_nt_query_t q = fn_l22_nt_query()) {
        alignas(8) unsigned char buf[1024] = {};
        ULONG got = 0;
        if (q(2 /* SystemPerformanceInformation */, buf, (ULONG) sizeof(buf), &got) >= 0 && got >= sizeof(fn_l22_sys_perf)) {
            c.sys_page_reads = ((const fn_l22_sys_perf *) buf)->PageReadCount;
        }
        if (!g_standby_denied.load(std::memory_order_relaxed)) {
            fn_l22_mem_list ml = {};
            ULONG got2 = 0;
            if (q(80 /* SystemMemoryListInformation */, &ml, (ULONG) sizeof(ml), &got2) >= 0) {
                uint64_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    s += (uint64_t) ml.PageCountByPriority[i];
                }
                c.standby = s*page;
            } else {
                g_standby_denied.store(true, std::memory_order_relaxed);
            }
        }
    }
#else
#    if !defined(__wasi__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        c.faults = (uint64_t) ru.ru_majflt + (uint64_t) ru.ru_minflt;
    }
#    endif
#    if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long av = sysconf(_SC_AVPHYS_PAGES);
    const long pg = sysconf(_SC_PAGESIZE);
    if (av > 0 && pg > 0) {
        c.avail_ram = (uint64_t) av*(uint64_t) pg;
    }
#    endif
#endif
    return c;
}

void llama_fn_l22_log(const char * kind, const char * fmt, ...) {
    if (llama_fn_l22_passlog_level() < 1) {
        return;
    }
    char text[2048];
    text[0] = '\0';
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(text, sizeof(text), fmt, ap);
        va_end(ap);
    }
    size_t len = strlen(text);
    while (len > 0 && (text[len - 1] == '\n' || text[len - 1] == '\r')) {
        text[--len] = '\0'; // one line: the caller's newline is not doubled
    }
    LLAMA_LOG_WARN("[TAG_FN_L22_PASSLOG] kind=%s %s\n", kind ? kind : "?", text);
}

//
// llama-ext.h
//

bool llama_fn_pin_on_oom_ext(const llama_model * model, size_t need_bytes) {
    GGML_UNUSED(model);
    return llama_fn_pin_on_oom(need_bytes);
}

bool llama_fn_pin_active_ext(void) {
    return llama_fn_pin_active();
}
