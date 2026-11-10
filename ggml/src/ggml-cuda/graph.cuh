#pragma once

// [TAG_R39_SYNC] upstream #32001 moved the CUDA graph and concurrent stream state from common.cuh into this file.
// The fork keeps its own graph state in common.cuh (ggml_cuda_graph with GRAPH2 / GRAPHKEEP / graph keys / chunk merge,
// the inline ggml_backend_cuda_context::cuda_graph() with [TAG_R34_SWEEPKEEP] and the by-value fn_l22_eager graph), so
// upstream's definitions are not taken; this header only keeps the include path of upstream files that name it.

#include "common.cuh"
