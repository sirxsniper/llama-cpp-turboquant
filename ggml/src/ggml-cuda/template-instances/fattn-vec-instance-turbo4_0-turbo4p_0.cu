// Mixed KV: turbo4 K + turbo4p V
//
// The two layouts hold identical values, so this pair costs no accuracy - it exists so K
// and V can be repacked independently.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4P_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4P_0);
