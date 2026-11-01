#pragma once

#include "gguf.h"
#include "llama.h"
#include "llama-arch.h"

#include <vector>

struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const bool gguf_ctx_owned = true;
    const struct llama_model * model;
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model * model);
    // [TAG_R34_SYNC] kept for the fork's MTP fixtures in test-llama-archs (upstream #30880 dropped it with its test-side
    // writer): writes arch-resolved keys into gguf_ctx (not owned), or into an own context when gguf_ctx is null
    llama_model_saver(enum llm_arch arch, struct gguf_context * gguf_ctx);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, uint64_t     value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
    void save(FILE * file); // [TAG_R34_SYNC] see above
};
