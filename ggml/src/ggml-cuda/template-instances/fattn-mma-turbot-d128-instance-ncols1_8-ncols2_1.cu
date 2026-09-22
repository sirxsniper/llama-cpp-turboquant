// Fork-specific instance, NOT produced by generate_cu_files.py.
//
// [TAG_TURBOT_ANY_D128] turbot tiered KV cache read path (docs/turbot/SPEC.md 7.1, 7.6), D=128. One file per
// (ncols1, ncols2) pair the D=128 case ladder in fattn.cu can select (ncols 8, 16, 32, 64), split like the 20 D=256
// fattn-mma-turbot-instance-*.cu files, so turbot-tables.cuh and the turbot kernels never enter an existing
// fattn-mma-f16 instance TU. The upstream generator does not know these files; maintain them by hand.
// -DGGML_CUDA_FA_TURBOT_D128=OFF drops them from the build (ggml-cuda/CMakeLists.txt).

#include "../fattn-turbot.cuh"

DECL_FATTN_TURBOT_CASE_D(128, 8, 1);
