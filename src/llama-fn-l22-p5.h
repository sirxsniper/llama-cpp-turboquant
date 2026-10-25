#pragma once

// [TAG_FN_L22_RESSET] round 22, P5 (RAM residency): declarations shared by llama-fn-tier.cpp, llama-fn-resident.cpp and
// llama-fn-dio.cpp only. The public residency / direct-I/O interface is in llama-fn-l22.h.
//
// qwen4exp on Windows with the warm pass on (LLAMA_FN_L15_WARM, tier off). Switches (profile, then environment):
//   LLAMA_FN_L22_RESSET=1    the residency table drives the warm pass (only what the prompt stream reads from the host,
//                            minus pinned and SSD experts, in stream order), host copies of VRAM-held experts go to the
//                            standby list, an expert that leaves VRAM is prefetched
//   LLAMA_FN_L22_REWARM=1    after >= 1 s without a prompt or a decode, the streamed pages that left RAM are read again
//                            (low priority, within the available RAM - 6 GiB, stops at the next decode)
//   LLAMA_FN_L22_SSDTIER=1|2 the least-routed cold experts that do not fit in RAM are read by direct I/O in prompts
//                            (LLAMA_FN_L22_SSDTIER_IO reader threads, default 8); 2 also lets the stream read layers
//                            whose pages are mostly missing that way (the stream's rule)
// The table is also built for LLAMA_FN_L22_PIN (the pin planner marks its experts in it). Nothing here changes a value
// the model computes.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct llama_model;
struct llama_model_loader;

// at the tier build (the loader still exists): the direct-I/O regions of the routed experts (SSDTIER >= 1)
void llama_fn_p5_on_build(const llama_model & model, llama_model_loader & ml);

// the warm worker, before its pass: the table, the pin planner's start (llama_fn_pin_on_ready), the SSD pre-plan, the
// service thread. false: no table (every switch off, or nothing to track)
bool llama_fn_p5_ready(const llama_model & model);

// the table drives the warm pass (RESSET)
bool llama_fn_p5_resset(const llama_model & model);

// [TAG_FN_L23_PINFIX] the warm pass has ended (llama_fn_p5_after_warm ran), or there is no table: the pin thread waits for it
bool llama_fn_p5_warm_done(const llama_model & model);

// the warm pass's floor of available RAM (LLAMA_FN_L15_WARM_FREE_MIB, raised to LLAMA_FN_L22_PIN_FLOOR_MIB + 1 GiB while
// pins are planned, R-P6-3); 0 when no table
size_t llama_fn_p5_floor(const llama_model & model);

// [TAG_FN_L22_RWIDTH] the bytes of the SSD set now (the experts the plan reads by direct I/O because the streamed set does
// not fit in RAM above the floor): > 0 = the machine is short of RAM; 0 when no table or no SSD set
size_t llama_fn_p5_ssd_bytes(const llama_model & model);

// one warm piece: experts [e0, e1) of part `part` (0 up, 1 gate, 2 down) of layer il; [p, p + n) their bytes
struct llama_fn_p5_piece {
    int          il;
    int          part;
    int          e0;
    int          e1;
    const char * p;
    size_t       n;
};

// the warm list in stream order (pieces of at most max_bytes) and, per trunk layer il, the index one past its last
// piece (layer_end[il]); *pin_pending: bytes of pinned experts not in RAM yet; *skipped: host bytes left out (VRAM, PIN,
// SSD). Empty when no table
std::vector<llama_fn_p5_piece> llama_fn_p5_warm_list(const llama_model & model, size_t max_bytes, std::vector<size_t> & layer_end,
        size_t * pin_pending, size_t * skipped);

// the ranges of a piece that still need the warm pass (the flags re-read now): appended to out, returns their bytes
size_t llama_fn_p5_piece_ranges(const llama_model & model, const llama_fn_p5_piece & pc,
        std::vector<std::pair<const char *, size_t>> & out);

// after the warm pass: the standby drop, the SSD post-adjust, the ready log line
void llama_fn_p5_after_warm(const llama_model & model, size_t warmed, double secs);

// the model goes away (tier deleter, before the mappings close): the service thread and the readers stop, the table goes
void llama_fn_p5_free(const llama_model & model);

// llama-fn-tier.cpp: activity of the decode / prompt paths for the idle re-warm. busy: a small decode or a prompt scope
// runs now; *last_us: the steady-clock time (us) of the last begin / end / stream layer
bool    llama_fn_p5_busy(int64_t * last_us);
int64_t llama_fn_p5_now_us();
// the activity stamps above are kept only while on (one relaxed load otherwise)
void    llama_fn_p5_track(bool on);

// llama-fn-dio.cpp: the reader statistics (bytes delivered, reads, failures) for the residency log
void llama_fn_p5_dio_stats(uint64_t * bytes, uint64_t * reads, uint64_t * fails);
// the model's reader goes away
void llama_fn_p5_dio_free(const llama_model & model);
