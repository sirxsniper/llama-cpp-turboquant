#pragma once

// [TAG_MOE_BRIDGE] per-context MoE host bridge (E:/turbot-gates/flashnext/PLAN.md SP-5, ggml-moe-bridge.h).
//
// With LLAMA_MOE_BRIDGE=1, a decode graph (T <= LLAMA_MOE_BRIDGE_MAX_T) no longer splits at the host-resident expert
// layers: each such layer posts x / ids / w to a mapped pinned channel, the device goes on with the hot experts and
// the shared expert, and a host executor computes the cold experts on the CPU MoE pool (ggml_cpu_moe_run, weighted
// sum) meanwhile; the device then waits (bounded) and adds. One device graph per step, no host sync per layer.
//
// Switches (all off / default unless set):
//   LLAMA_MOE_BRIDGE=1                    enable (one bridge per process: the first context with host expert layers)
//   LLAMA_MOE_BRIDGE_WAIT=spin|hostfunc   device spin on a mapped flag + executor thread (default), or a host function
//                                         node that runs the job on the driver thread
//   LLAMA_MOE_BRIDGE_SPIN_US=2000         executor and pool workers spin this long after a job, then sleep
//   LLAMA_MOE_BRIDGE_TIMEOUT_MS=500       a device wait gives up after this long if the host has not taken the job
//                                         ([TAG_FN_R1_BRIDGE_RETRY] was 50, and a cold start missed it). The context
//                                         then rolls the ubatch back and computes it again without the bridge when the
//                                         memory can drop the whole ubatch (attention caches always; a recurrent state
//                                         with [TAG_FN_R1_BRIDGE_RB]), else the ubatch fails as below
//   LLAMA_MOE_BRIDGE_RB=0                 [TAG_FN_R1_BRIDGE_RB] (on by default with speculative decoding on a hybrid
//                                         model) off: no extra ring token, the bridge keeps graphs up to MAX_T, and a
//                                         failed bridged verify ubatch fails (its recurrent state cannot roll back)
//   LLAMA_MOE_BRIDGE_PF=1                 [TAG_FN_R2_BRIDGE_PF] (spin mode) after each layer's job the pool's workers
//                                         predict the next layer's experts (its router on this layer's input, top
//                                         LLAMA_MOE_BRIDGE_PF_K=12 per token), skip the hot ones and pull the rest into
//                                         the CPU caches while the device runs that layer's attention (DRAM is idle
//                                         then); the next job stops them. Never changes a value.
//                                         LLAMA_MOE_BRIDGE_PF_MODE=0 real loads (default), 1 software prefetches
//                                         LLAMA_MOE_BRIDGE_PF_SOLO=1: the executor stays out of the compute, so every
//                                         worker computes exactly what it prefetched (no un-prefetched straggler)
//   LLAMA_MOE_BRIDGE_SYNC=1               [TAG_FN_R2_BRIDGE_SYNC] (default) while the bridge is paused or off (the
//                                         retried ubatch, the 16 steps after a deadline miss, after 3 errors), the
//                                         layers it takes run its host job as a CPU graph op (ggml_moe_host_sum) and
//                                         sum like the bridged graph: the same values, so a stall never changes the
//                                         output. 0: the plain CPU split (the slots summed in another float order)
//   LLAMA_MOE_BRIDGE_RB_TOKENS=1          [TAG_FN_R2_BRIDGE_NOSPEC] a hybrid model without speculative decoding has no
//                                         ring at all: the context gets a ring of this many tokens (1..8) and the bridge
//                                         takes graphs of at most that many tokens, so every bridged ubatch rolls back
//   LLAMA_MOE_BRIDGE_JOB_MAX_MS=1000        ... or after this long once it has (slow job); either way the ubatch fails
//                                         (and is computed again as above) and the bridge pauses 16 steps; 3 errors
//                                         within 1024 bridged graphs turn it off for the context
//   LLAMA_MOE_BRIDGE_MAX_T=8              largest graph width that uses it (1..16)
//   LLAMA_MOE_BRIDGE_THREADS=<n>          pool threads including the executor (default: the context's n_threads)
//   LLAMA_MOE_BRIDGE_CPUMASK=<hex>        pool CPUs (default: one per physical core, except the first core)
//   LLAMA_MOE_BRIDGE_PRIO=<0..3>          pool thread priority (default 2, high)
//   LLAMA_MOE_BRIDGE_STATS=1              job and wait statistics every 256 bridged graphs
//   LLAMA_MOE_BRIDGE_TEST_STALL=<ms>      test (spin): the executor sleeps this long after job 199 (and every
//   LLAMA_MOE_BRIDGE_TEST_STALL_EVERY=<n>   n jobs after it) before it takes the next, to exercise the timeout path
//   LLAMA_MOE_BRIDGE_DMA=1                [TAG_FN_R4_BRIDGE_DMA] with LLAMA_MOE_DMA_SHARE=<0..1>|auto (llama-moe-gen5.h):
//                                         the DMA share inside the bridged graphs - the GPU computes its hot experts, then
//                                         the fetched ones (copied from the pinned ring by SM loads) when they land, the CPU
//                                         pool the rest; share = auto moves the split by the measured rates
//
// Lever round 3, CPU side (qwen4exp only; off by default; every one leaves the values bit for bit as they are):
//   LLAMA_MOE_POOL_STATS=1                [TAG_FN_L3_CPU_STATS] the pool's job anatomy every 256 bridged graphs: bytes,
//                                         GB/s, start lag, gate/up and down time per thread, barrier waits and spread,
//                                         per-L3-domain GB/s, the prefetch's window and bytes, page faults per job, the
//                                         slowest workers (CPU, GB/s), the slowest job, the wait for a stopped prefetch
//   LLAMA_MOE_POOL_EXEC_CPU=<cpu>         [TAG_FN_L3_CPU_PLACE] (default: the first core of the pool) the executor's CPU,
//                                         e.g. 1 = the SMT sibling of core 0 (the main thread): every pool core computes
//   LLAMA_MOE_POOL_PF_STREAMS=<n>         [TAG_FN_L3_CPU_PFSTREAMS] (default 1; with split stable / steal) each worker pulls
//                                         n regions of an expert at once, one line of each in turn (more misses in flight)
//   LLAMA_MOE_DMA_FILL_GAP=1              [TAG_FN_L3_CPU_FILL] (with the DMA share) the ring fills pause while a pool job
//                                         reads DRAM and run between the jobs (placement noise: the ring-ready set moves)
//   LLAMA_MOE_DMA_FILL_CPUS=<hex mask>    [TAG_FN_L3_CPU_FILL] the ring fill threads on these CPUs only (e.g. 0xA = 1, 3:
//                                         the SMT siblings of the main thread's and the executor's cores)
//   LLAMA_MOE_POOL_SPLIT=range|stable|steal [TAG_FN_L3_CPU_SPLIT] (default range) stable: piece p of expert e goes to
//                                         worker (h(e) + p) mod n in the job and in the prefetch, and the prefetched
//                                         experts go first, so each core computes what it pulled; steal: stable, then
//                                         idle workers take free pieces from the end of the others' lists
//   LLAMA_MOE_POOL_SWPF=<lines>           [TAG_FN_L3_CPU_SWPF] (default 0) each worker software-prefetches this many cache
//                                         lines at every 4 KiB page of its next piece before it computes the current one
//                                         (the hardware stream prefetcher stops at every page of the file mapping)
//   LLAMA_MOE_BRIDGE_PF_RANK=1            [TAG_FN_L3_CPU_PFRANK] the host-router prefetch pulls by rank (every token's
//                                         most likely expert first) instead of in expert order, so a stopped prefetch
//                                         holds the likeliest experts
//   LLAMA_MOE_BRIDGE_PF_DEV=1|2           [TAG_FN_L3_CPU_DEVPRED] (with LLAMA_MOE_BRIDGE_PF=1) the device predicts: after
//                                         each post the graph applies the next layer's router to this layer's input
//                                         (top LLAMA_MOE_BRIDGE_PF_K per token) and writes the ids to the channel; the
//                                         executor prefetches from them (by rank) with no router on the host and no
//                                         f16 router copies (-117 MiB RAM); LLAMA_MOE_BRIDGE_PF_DEV_WAIT_US=50 bounds the
//                                         wait for a hint that is late. 2: the router's input is the next layer's own FFN
//                                         mixer (hyper-connection norm and gates) applied to this layer's residual, a
//                                         closer stand-in for that layer's input (a few more small kernels after the post)

#include <cstdint>

struct llama_model;
struct ggml_tensor;
struct llama_moe_bridge;

// nullptr when disabled or not possible here (the reason is logged). max_t_cap > 0: graphs wider than that never take
// the bridge ([TAG_FN_R1_BRIDGE_RB]: the widest ubatch the recurrent ring can roll back whole). seq_t_cap > 0: nor graphs
// with more tokens of one sequence ([TAG_FN_L13_BRIDGE_SEQS]: the ring is per sequence, so 2 streams of 3 tokens fit)
llama_moe_bridge * llama_moe_bridge_create(const llama_model & model, int n_threads, int max_t_cap = 0, int seq_t_cap = 0);
void               llama_moe_bridge_free(llama_moe_bridge * br);
// [TAG_FN_L4_EXIT] teardown only: every device wait and fetch of this bridge gives up at once (no CUDA call), so no graph
// of its context spins on mapped flags while the context synchronizes and frees; the bridge is unusable afterwards
void               llama_moe_bridge_release(llama_moe_bridge * br);
// [TAG_FN_R2_BRIDGE_NOSPEC] LLAMA_MOE_BRIDGE=1 and the model has a layer the bridge could take (cheap, no pool, no device)
bool               llama_moe_bridge_wanted(const llama_model & model);
// [TAG_FN_R2_BRIDGE_SYNC] the bridge exists, is paused or off, and LLAMA_MOE_BRIDGE_SYNC is on
bool               llama_moe_bridge_sync(const llama_moe_bridge * br);
// [TAG_FN_L3_CPU_DEVPRED] predicted ids per token a graph writes after each post (0: no hints), and the input of the
// prediction: 1 = this layer's FFN input, 2 = the next layer's FFN mixer on this layer's residual (0: no hints)
int                llama_moe_bridge_hint_k(const llama_moe_bridge * br);
bool               llama_moe_bridge_hint_fused(const llama_moe_bridge * br); // [TAG_FN_L9_HINTTOPK]
int                llama_moe_bridge_hint_mode(const llama_moe_bridge * br);

// [TAG_FN_R4_BRIDGE_DMA] LLAMA_MOE_BRIDGE_DMA=1 (with LLAMA_MOE_DMA_SHARE=<share>|auto): after llama_moe_gen5_init (bridge
// mode) and before the reserve, register the DMA ring with the device side; the bridged graphs then fetch a share of
// each job's experts over PCIe (GGML_OP_MOE_HOST_FETCH) while the CPU computes the rest. false: no DMA share (logged).
bool               llama_moe_bridge_attach_dma(llama_moe_bridge * br, const void * owner);
bool               llama_moe_bridge_dma(const llama_moe_bridge * br);

// graph side. active: graphs built now may use it (not paused after an error, not disabled)
bool llama_moe_bridge_active(const llama_moe_bridge * br);
int  llama_moe_bridge_max_t (const llama_moe_bridge * br);
// [TAG_FN_L13_BRIDGE_SEQS] a graph of n_tokens whose largest sequence has n_seq_tokens of them may take the bridge
bool llama_moe_bridge_fits  (const llama_moe_bridge * br, int n_tokens, int n_seq_tokens);
int  llama_moe_bridge_n_used(const llama_moe_bridge * br);
// the bridge id and channel for this layer's up_exps, false if the layer is not bridged
bool llama_moe_bridge_layer (const llama_moe_bridge * br, const ggml_tensor * up_exps, int32_t * id, int32_t * chan);

// [TAG_FN_L3_MTP_COST2] cold-expert counts for the MTP draft-length policy (SPEC_MTP_COST=2). With tracking on, each
// bridged graph sums over its layers, per token prefix t, the distinct experts of tokens 0..t that are not in the VRAM
// hot set. llama_moe_bridge_last: the last bridged graph that completed without error - its width, its counts
// (cold[t] for t < min(width, n)) and its serial number (one per bridged graph). Owner thread only.
constexpr int LLAMA_MOE_BRIDGE_TRACK_T = 16;
void llama_moe_bridge_track(llama_moe_bridge * br, bool on);
bool llama_moe_bridge_last (const llama_moe_bridge * br, int32_t * n_tokens, uint32_t * cold, int32_t n, uint64_t * serial);

// runtime, on the owning context's thread
void llama_moe_bridge_step (llama_moe_bridge * br);            // before the graph parameters: re-arm after a pause
void llama_moe_bridge_begin(llama_moe_bridge * br, bool used); // before the graph runs: wake the executor, or park it
// after a graph with used = true completed (also a failed one): false = its output is invalid. Jobs that come after it
// (their wait timed out) are stale: they return zeros without running and without the routing observer.
bool llama_moe_bridge_end  (llama_moe_bridge * br);

// [TAG_FN_L3_HOST_QUIET] LLAMA_FN_HOST_QUIET=1 (qwen4exp): background host copies (the hot set's uploads, the DMA
// ring's fills) wait here while a bridged graph runs, so its CPU experts get the DRAM bandwidth, and copy between
// graphs. Returns at once when no bridge has the switch on; a wait ends after max_ms at the latest.
void llama_moe_bridge_quiet_wait(int max_ms);
