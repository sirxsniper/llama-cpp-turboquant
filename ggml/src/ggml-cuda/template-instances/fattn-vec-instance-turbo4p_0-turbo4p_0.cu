// TurboQuant4P CUDA flash attention vec kernel instantiation
//
// No D=64: a turbo4p head must be a whole number of 128-element WHT groups, so a 64-wide
// head would put two heads under one norm. The kernel static_asserts on it.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4P_0, GGML_TYPE_TURBO4P_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4P_0, GGML_TYPE_TURBO4P_0);
