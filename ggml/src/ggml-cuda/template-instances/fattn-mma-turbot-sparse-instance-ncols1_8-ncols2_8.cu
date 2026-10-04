// Fork-specific instance, NOT produced by generate_cu_files.py.
//
// [TAG_FN_TURBOT_SPARSE] sparse turbot FA (qwen4exp QSA layers): gathers the cells of one index list per query tile.
// Maintained by hand, like the other fattn-mma-turbot-instance-*.cu files.

#include "../fattn-turbot.cuh"

DECL_FATTN_TURBOT_SPARSE_CASE(8, 8);
