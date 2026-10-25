#pragma once

// [TAG_FN_L22_PIN] moe-bridge-probe test 7 (moe-bridge-pin.cpp): pinned file pages. Arguments from main
// (moe-bridge-probe.cu); the output goes through the probe's own log function.

#include <cstddef>
#include <string>
#include <vector>

struct pin_probe_args {
    std::vector<std::string> files;                       // GGUF shards with routed experts (mapped read-only)
    std::vector<double>      ladder_gib = { 4, 8, 16, 32 };
    int                      hot_every  = 0;              // > 0: every n-th expert is "VRAM-held" (holes like real use)
    size_t                   full_mib   = 0;              // allocation tests (0 = skip), sizes from the NEW build's log
    size_t                   borrow_mib = 0;
    size_t                   vision_mib = 0;
    size_t                   lock_mib   = 1024;
    size_t                   margin_mib = 8192;           // NON_LOCAL margin
    size_t                   floor_mib  = 6144;           // RAM floor
    bool                     repeat     = true;           // step 5: the ladder once more
    bool                     mapped     = true;           // register with cudaHostRegisterMapped too (the server does with DMASHARE)
    size_t                   range_mib  = 256;            // the server's cut (LLAMA_FN_L22_PIN_RANGE_MIB), 0 = whole runs
};

using pin_say_fn = void (*)(const char * fmt, ...);

// 0: done (see the verdict line), 1: FAIL
int moe_bridge_pin_test(const pin_probe_args & a, pin_say_fn say);

// [TAG_FN_L23_PINFIX] --only bounce: 0 PASS, 1 FAIL
int moe_bridge_bounce_test(pin_say_fn say);
