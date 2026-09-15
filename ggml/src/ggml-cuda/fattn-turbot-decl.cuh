#pragma once

// [TAG_TURBOT] Host-only declarations of the turbot flash attention read path for fattn.cu (docs/turbot/SPEC.md 7.1).
//
// Deliberately includes common.cuh only. The device code, the kernel template and the __constant__ codebook tables
// live in fattn-turbot.cuh, which only the 20 turbot template-instance TUs include, so no existing kernel TU (and not
// fattn.cu) can ever see turbot-tables.cuh.

#include "common.cuh"

// Flash attention over a turbot K/V cache: K and V are turbot views, src[7] the layer's young pool, src[8] the granule
// table. Defined in fattn-turbot.cuh, instantiated for D = 256 only.
template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_turbot_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define DECL_FATTN_TURBOT_CASE(ncols1, ncols2) \
    template void ggml_cuda_flash_attn_ext_turbot_case<256, 256, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// Exactly the (ncols1, ncols2) pairs ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<256, 256, ncols2> can select for
// ncols2 in {1, 2, 4, 8}: ncols 8, 16, 32, 64 and the [TAG_FA_NCOLS_128] tier. One instance file per pair in
// template-instances/fattn-mma-turbot-instance-ncols1_<n1>-ncols2_<n2>.cu.
extern DECL_FATTN_TURBOT_CASE(  8, 1);
extern DECL_FATTN_TURBOT_CASE(  4, 2);
extern DECL_FATTN_TURBOT_CASE(  2, 4);
extern DECL_FATTN_TURBOT_CASE(  1, 8);

extern DECL_FATTN_TURBOT_CASE( 16, 1);
extern DECL_FATTN_TURBOT_CASE(  8, 2);
extern DECL_FATTN_TURBOT_CASE(  4, 4);
extern DECL_FATTN_TURBOT_CASE(  2, 8);

extern DECL_FATTN_TURBOT_CASE( 32, 1);
extern DECL_FATTN_TURBOT_CASE( 16, 2);
extern DECL_FATTN_TURBOT_CASE(  8, 4);
extern DECL_FATTN_TURBOT_CASE(  4, 8);

extern DECL_FATTN_TURBOT_CASE( 64, 1);
extern DECL_FATTN_TURBOT_CASE( 32, 2);
extern DECL_FATTN_TURBOT_CASE( 16, 4);
extern DECL_FATTN_TURBOT_CASE(  8, 8);

extern DECL_FATTN_TURBOT_CASE(128, 1);
extern DECL_FATTN_TURBOT_CASE( 64, 2);
extern DECL_FATTN_TURBOT_CASE( 32, 4);
extern DECL_FATTN_TURBOT_CASE( 16, 8);
