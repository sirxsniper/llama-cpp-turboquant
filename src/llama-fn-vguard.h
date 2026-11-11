#pragma once

// [TAG_R40_VGUARD] the WDDM paging guard (round 40). When another process allocates VRAM (Kyro's GGUF embedder, a game, a
// browser) while the card is full, Windows' video memory manager demotes 384-448 MiB of the server's VRAM to system memory
// for 5-14 s and a fresh prompt crawls (1,070-1,500 instead of 2,300-2,600 t/s on 32K; r39 stall/resv). The guard reads the
// process's own WDDM counters (D3DKMTQueryStatistics, ~1 us) where the owner thread may move hot-set VRAM, and hands whole
// KV-lend slot groups of the hot set back to the driver (the "vg" share, llama-moecache.h) so the room is made by us instead
// of by paging:
//   H = Budget - Usage (our LOCAL segment group), F = CommitLimit - BytesResident (the local segment, all processes),
//   D = Usage - our resident bytes - the calm baseline (our bytes demoted to system memory)
//   ACT      H < HACT or F < FACT at a FULL switch / an encode: hold groups up to TARGET (H >= HTGT and F >= FTGT) before
//            the tail goes (groups short: that prompt runs at the base width), the share returns at SMALL
//   DEMOTED  D >= DEM on 2 polls >= 20 ms apart; PRESSURE  Budget >= DROP below the last calm Budget while below TARGET:
//            at decode entry (synced) or a prefill-stream gate (device idle, release only) the share grows
//            (max(2 groups, D + 1 group), doubled per further second of demotion; or the budget drop up to TARGET)
//   regrow   decode entry in SMALL: one group per 2 s once D <= 16 for 3 s, no PRESSURE for CALM_S, H - group >= 512 MiB
// Windows only (a CUDA device with a LUID, SupportsDemotion, the queries working, the slot lend on virtual memory); every
// other build or model (the 27B, Linux, Vulkan, HIP, MUSA) logs "off" once and each hook is one bool test.
//   LLAMA_R40_VGUARD=1 (default) on, 0 = off (no KMT call, no pool pre-grow: v0.30.0), 2 = observe (poll + log only)
//   LLAMA_R40_VGUARD_HACT 2304, _FACT 2560, _HTGT 2944, _FTGT 3200, _DEM 64, _DROP 256, _MAX 2048 (MiB), _POLL_MS 20,
//   _CALM_S 10, _REACT 1 (0: switch / encode rule only), _PREGROW 1 (the FULL pool's growth committed at the switch),
//   _CHECK 0 (1: assert no stream D2D source lies in a KV group), _FORCE 0 (tests: act at every switch / gate / 8th decode)

#include <cstddef>
#include <cstdint>

struct ggml_backend_device;

struct llama_fn_vg_hooks {
    void * user = nullptr;
    void (*sync)(void * user) = nullptr; // every graph of the model done, the hot step joined (the owner's r35_hook_sync)
    bool (*idle)(void * user) = nullptr; // no stream of any context of the model busy, no hot step running (never waits)
};

int  llama_fn_vg_mode();  // LLAMA_R40_VGUARD: 0 off, 1 on, 2 observe (read once)
bool llama_fn_vg_check(); // LLAMA_R40_VGUARD_CHECK=1 (read once)

// the hot set's owner, once after its fit: true = armed for this process (one owner). why_not: off for that reason (one
// INFO line per process, no KMT call). dev: the hot set's device (its adapter LUID through the CUDA proc)
bool llama_fn_vg_arm(const void * owner, ggml_backend_device * dev, const llama_fn_vg_hooks & hooks, const char * why_not);

// hooks, owner thread only; each returns at once when the guard is not armed
bool   llama_fn_vg_switch_pre();         // cbuf_set(true) after its synchronize, before the tail release: true = run the
                                         // FULL state at the base width (the groups did not cover the need)
size_t llama_fn_vg_pregrow_bytes();      // after the FULL reserve: grow the compute pool to this now (0: nothing)
void   llama_fn_vg_full_hwm(size_t hwm); // at the return to SMALL: the pools' high-water of the FULL period that ends
void   llama_fn_vg_small();              // after the return to SMALL: the share taken at a switch / encode comes back
void   llama_fn_vg_decode(bool small);   // decode() entry, after the idle-slot share's batch work
void   llama_fn_vg_encode();             // before the projector's take (an image encode)
void   llama_fn_vg_gate();               // a prefill-stream gate (ith 0, the decode thread between two layers)
