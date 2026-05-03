// TurboQuant InnerQ host-side state — definitions live here in ggml-base
// so they're available to both ggml-cuda.dll (writer) and llama.dll (reader)
// regardless of whether GGML_BACKEND_DL is on (dynamic CUDA backend) or off
// (CUDA backend statically linked).

#include "ggml-turbo-innerq.h"

extern "C" {

bool g_innerq_finalized = false;

float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
};

static bool g_innerq_tensor_needs_update = false;

void turbo_innerq_publish(const float * scale_inv, int group_size) {
    for (int i = 0; i < group_size && i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = scale_inv[i];
    }
    for (int i = group_size; i < INNERQ_MAX_CHANNELS; i++) {
        g_innerq_scale_inv_host[i] = 1.0f;
    }
    g_innerq_finalized = true;
    g_innerq_tensor_needs_update = true;
}

bool turbo_innerq_needs_tensor_update(void) {
    return g_innerq_tensor_needs_update;
}

void turbo_innerq_mark_tensor_updated(void) {
    g_innerq_tensor_needs_update = false;
}

}
