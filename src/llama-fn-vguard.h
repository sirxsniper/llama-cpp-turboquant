#pragma once

// [TAG_R40_VGUARD] the WDDM paging guard (round 40, R40-VGUARD-2). When another process allocates VRAM (Kyro's GGUF embedder,
// a game, a browser) while the server runs a fresh prompt, Windows' video memory manager demoted part of the server's VRAM
// to system memory for seconds and the prompt crawled (r39 stall/resv). Iteration 1 showed that the memory demoted was the
// memory a FULL switch had just created (20 of 20 demoted prompts began -0.4..2.9 s from the switch, with 2.9-3.6 GB of
// the card free): v0.30.0 freed 8 GiB at a switch and created ~8 GiB of new WDDM allocations. Two parts:
//
// ZERO-COMMIT (LLAMA_R40_ZC, default 1): the hot set's tail and borrow handles move into the FULL compute reserve and the
//   compute pool instead of being freed and created again (ggml-cuda's ggml_cuda_zc: cuMemRetainAllocationHandle +
//   cuMemUnmap into a stash, cuMemMap at a fixed arena address), and back into the tail at SMALL; a KV-cache growth and the
//   projector's take move slot-group handles the same way. What a step cannot cover comes from new memory (logged).
//
// THE GUARD (poll + share): it reads the process's WDDM counters where the owner thread may move hot-set VRAM (D3DKMT ~0.3 us,
//   PDH ~1.9 us, at most every POLL_MS) and hands whole KV-lend slot groups of the hot set back to the driver (the "vg" share):
//   H = Budget - Usage (our LOCAL segment group), F = CommitLimit - BytesResident (the local segment, all processes)
//   D = Usage - our PDH "GPU Process Memory(pid_*_luid_*_phys_0)\Dedicated Usage" - the calm offset at arm (our bytes
//       demoted to system memory); detection arms only when |D| <= 32 MiB at arm
//   deficit = solo budget - Budget (solo: the budget at arm, raised when the budget rises, never lowered by an action)
//   PRESSURE  deficit >= DROP: at decode entry / a prefill-stream gate / a switch the share holds at least the deficit
//   FULL switch / encode: PRESSURE, or H < HTGT or F < FTGT -> groups until H >= HTGT and F >= FTGT. A zero-commit switch
//            commits nothing, so its H/F levels are ZHTGT / ZFTGT (the measured desktop sat at H 2794-2808 / F 3174 at the
//            first switch of every undisturbed prompt: HTGT 2944 / FTGT 3200 would hold a group for good)
//   DEMOTED  D >= DEM on 2 polls >= 20 ms apart: always logged; REACT=1 also gives groups until F >= REACT_F
//   regrow   one group per REGROW_S, only in a SMALL generation step without prompt rows, >= QUIET_S after the last batch
//            with prompt rows, after deficit < DROP and D <= 16 for CALM_S (or, with the other process still there, its
//            deficit steady within 64 MiB for CALM_S and the share >= the deficit after the group), while H - group and
//            F - group stay at the levels
// Windows only (a CUDA device with a LUID, SupportsDemotion, the queries working, the slot lend on virtual memory); every
// other build or model (the 27B, Linux, Vulkan, HIP, MUSA) logs "off" once and each hook is one bool test.
//   LLAMA_R40_VGUARD=1 (default) on, 0 = off (no WDDM call, no zero-commit: v0.30.0), 2 = observe (poll + log, zero-commit on)
//   LLAMA_R40_ZC=1 (default; 0: the v0.30.0 switch with the guard still on, for A/B runs)
//   LLAMA_R40_VGUARD_HTGT 2944, _FTGT 3200, _ZHTGT 2304, _ZFTGT 2560, _DEM 64, _DROP 256, _MAX 2048, _REACT_F 3800 (MiB),
//   _POLL_MS 20, _CALM_S 10, _QUIET_S 2, _REGROW_S 2, _PRESS 1 (0: no PRESSURE share), _REACT 0 (1: the DEMOTED reaction),
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
bool llama_fn_vg_switch_pre();                      // cbuf_set(true) after its synchronize, before the tail release: true =
                                                    // run the FULL state at the base width (the groups did not cover the need)
void llama_fn_vg_decode(bool small, bool gen_only); // decode() entry; gen_only: no prompt row in the batch (every row an output)
void llama_fn_vg_encode();                          // before the projector's take (an image encode)
void llama_fn_vg_gate();                            // a prefill-stream gate (ith 0, the decode thread between two layers)

// [TAG_R40_VGUARD] zero-commit glue (owner thread). _arm: once, the hot set's device with the compute-buffer switch; true =
// on (LLAMA_R40_VGUARD != 0, LLAMA_R40_ZC != 0, a Windows CUDA device with VMM). _open / _close: a scope around a release
// and the maps or buffers that take its handles (_open returns its id, 0 while off; _close(id) frees what the scope parked
// and nobody took, everything left when it was the last open one). _window: +1 / -1 around buffer allocations that may be
// built from parked handles. No-ops while off.
// _note: one line per FULL / SMALL switch (the first 8, then every 64th, and every switch that needed new memory)
bool llama_fn_zc_arm(ggml_backend_device * dev);
bool llama_fn_zc_on();
int  llama_fn_zc_open();
void llama_fn_zc_close(int id);
void llama_fn_zc_window(int delta);
void llama_fn_zc_note(bool full, double ms);
size_t llama_fn_zc_stash_bytes(); // the handles parked now (0 while off): free memory a FULL reserve can be built from
