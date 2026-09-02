// TurboQuant5P CUDA flash attention vec kernel instantiation [TAG_TURBO5P]
//
// No D=64, for the same reason as turbo4p: a head must be a whole number of 128-element
// WHT groups, so a 64-wide head would put two heads under one norm.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO5P_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO5P_0);
