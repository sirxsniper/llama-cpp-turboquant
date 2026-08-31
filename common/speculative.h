#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // optional sparse proposal distributions, one per draft token
    std::vector<common_speculative_token_dist> * dists = nullptr;

    float temperature = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// tell the speculative context how many prompt tokens still follow the ubatch that is
// about to be passed to common_speculative_process(); 0 during generation. Lets a
// sliding-window drafter skip prompt ubatches whose KV would be evicted unused.
void common_speculative_set_prefill_after(common_speculative * spec, int32_t n_after);

// [TAG_SPEC_PREFILL_TAIL_PER_SEQ] Per-sequence form of the above, which is what the skip
// decision must use. The scalar is a MAX over slots, so on its own it lets a prefilling slot
// speak for a generating one that merely shares the batch, and the skip wipes that slot's
// drafter KV. Call clear() then set() for each slot still processing a prompt.
void common_speculative_clear_prefill_after_seq(common_speculative * spec);
void common_speculative_set_prefill_after_seq(common_speculative * spec, llama_seq_id seq_id, int32_t n_after);

// true if any registered implementation actually reads dparams.prompt. Only the n-gram
// drafters do; the model-based ones (dflash, mtp, eagle3, simple) never look at it, so the
// server can skip materialising the whole prompt vector every step.
bool common_speculative_wants_prompt(const common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding

// [TAG_SPEC_PHASE_PROBE] Wall-clock accounting of one speculative decode step.
// Enable with SPEC_PHASE_PROBE=1; it prints a breakdown every 128 steps.
//
// History, so nobody re-opens this: the probe originally recorded 12.7 ms of speculative
// overhead at ~0 context against 38.9 ms at 131K, while plain decode only moved 18.1 ->
// 21.3 ms, and asked which phase scaled with context. That question is ANSWERED and the
// cause is FIXED. A verification batch is Q = n_draft+1 tokens wide, which is too wide
// for the FA-vec kernel (one pass, six query heads packed) and fell through to MMA, whose
// ncols2 ladder took the largest power of two DIVIDING gqa_ratio - 2 for gqa_ratio 6 -
// giving ntiles_z_gqa = 3 and three full passes over K and V per layer per step. Three
// passes over a cache whose size is linear in context is exactly a term that vanishes at
// empty context and dominates at 131K. Picking the packing by Q width instead (the
// FA_NCOLS2_MAXQ gate in ggml-cuda/fattn.cu) took step time at d131072 from 60.46 ms to
// 43.02 ms. The probe stays because it is still the right instrument, but the numbers
// above are historical, not an open problem.
enum common_spec_phase {
    COMMON_SPEC_PHASE_TGT_DECODE = 0, // llama_decode(ctx_tgt) + its synchronize
    COMMON_SPEC_PHASE_PROCESS,        // common_speculative_process: layer extract + drafter encode/inject
    COMMON_SPEC_PHASE_DRAFT,          // common_speculative_draft: drafter decode + lattice read
    COMMON_SPEC_PHASE_ACCEPT,         // common_speculative_accept
    COMMON_SPEC_PHASE_SAMPLE,         // target sampling / draft verification
    COMMON_SPEC_PHASE_COUNT
};
bool common_speculative_probe_enabled();
void common_speculative_probe_add(int phase, double ms);
void common_speculative_probe_step();

void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
