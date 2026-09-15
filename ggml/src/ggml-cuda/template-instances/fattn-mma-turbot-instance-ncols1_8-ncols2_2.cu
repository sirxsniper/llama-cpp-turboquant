// Fork-specific instance, NOT produced by generate_cu_files.py.
//
// [TAG_TURBOT] turbot tiered KV cache read path (docs/turbot/SPEC.md 7.1, 7.6), D=256 only. One file per
// (ncols1, ncols2) pair the D=256 case ladder in fattn.cu can select, so turbot-tables.cuh and the turbot kernels
// never enter an existing fattn-mma-f16 instance TU and every existing kernel keeps its codegen. The upstream
// generator does not know these files, the same as the [TAG_FA_NCOLS_128] instances; maintain them by hand.

#include "../fattn-turbot.cuh"

DECL_FATTN_TURBOT_CASE(8, 2);
