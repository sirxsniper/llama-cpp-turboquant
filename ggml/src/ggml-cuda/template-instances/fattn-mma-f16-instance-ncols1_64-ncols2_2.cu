// Fork-specific instance, NOT produced by generate_cu_files.py.
//
// [TAG_FA_NCOLS_128] ncols = 128 exists only for the D=256 deep-prefill tier. The
// upstream generator stops at ncols = 64 and is in any case stale against this fork:
// it does not know the turbo KV types or the 640/512 head size, so re-running it
// deletes the hand-maintained fattn-vec turbo instances. The turbo vec instances are
// already maintained by hand for the same reason.

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 64, 2);
