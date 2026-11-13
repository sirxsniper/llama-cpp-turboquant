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
//
// Round 22, CPU pool (qwen4exp only; off unless set; the first three never change a value):
//   LLAMA_MOE_POOL_STAGGER=<G>            [TAG_FN_L22_POOL_STAGGER] (split stable / steal) pool thread i walks a job's experts
//                                         from position (i % G)*n/G on, and pulls its prefetch from the same share on, so at
//                                         most ~n_threads/G threads fault one cold expert's pages at a time (0 = off, <= 16)
//   LLAMA_MOE_POOL_PRETOUCH=1             [TAG_FN_L22_PRETOUCH] one helper thread (LLAMA_MOE_POOL_PRETOUCH_CPU=<cpu>, default
//                                         an SMT sibling of a worker's core) reads one byte per page of the cold experts the
//                                         next jobs compute (the device's prediction for the next layer, a posted job's later
//                                         experts) that no job of their layer read within the last 64 jobs
//   LLAMA_MOE_POOL_JOBPF=1|2              [TAG_FN_L22_JOBPF] one read-ahead request (PrefetchVirtualMemory) per job, sent by a
//                                         helper thread at the post (no system call on the job's path): 1 its cold experts no
//                                         job of the layer read within the last 64 jobs, 2 all of them
//   LLAMA_FN_L22_X_STATS=1                [TAG_FN_L22_X_STATS] per job: the cold experts, and how many of them (and how much
//                                         routing weight) the rank rule (R = 6..9) and the weight rule (< 0.05, < 0.1) would
//                                         drop; one line per 4096 jobs and when the executor goes idle. Counters only
//   LLAMA_FN_L22_X_TAILSKIP=<R>           [TAG_FN_L22_X_TAILSKIP] CHANGES ANSWERS: in decode windows the graph flags
//   LLAMA_FN_L22_X_TAILSKIP_W=<w>         (GGML_MOE_BRIDGE_JOB_TAIL), a cold expert whose smallest rank over the window is
//                                         >= R, or whose largest weight is < w, contributes nothing (no renormalization);
//                                         the sync op of a retried ubatch drops the same experts. R >= n_used: weight rule
//                                         only. LLAMA_FN_L22_X_TAILSKIP_PF=1: the next-layer prefetch skips ranks >= R
//   LLAMA_FN_L22_X_DEFER=1|check          [TAG_FN_L22_X_DEFER] CHANGES ANSWERS (spin waits): in one-sequence decode windows the
//   LLAMA_FN_L22_X_DEFER_W=<w>            graph flags (GGML_MOE_BRIDGE_JOB_DEFER), the host answers a layer's heavy cold experts
//   LLAMA_FN_L22_X_DEFER_R=<R>            first and its light ones (largest weight < w, or best rank >= R; tail-skipped ones in
//                                         neither part) on the layer's defer channel; the graph adds them after the next
//                                         layer's token mixer. check: added in the same layer (plumbing test). Neither rule
//                                         set: w = 0.1. The sync op of a retried ubatch splits the same way
//   LLAMA_FN_L22_TBO=1|2                  [TAG_FN_L22_TBO] (lookup; spin waits; two sequences per graph fit) a second channel
//                                         per bridged layer for the two-chain decode graph (D5, llama-fn-tbo.h); its posts
//                                         carry GGML_MOE_BRIDGE_JOB_NOPF (no next-layer prefetch between the chains)

#include <cstdint>

struct llama_model;
struct ggml_tensor;
struct llama_moe_bridge;

// nullptr when disabled or not possible here (the reason is logged). max_t_cap > 0: graphs wider than that never take
// the bridge ([TAG_FN_R1_BRIDGE_RB]: the widest ubatch the recurrent ring can roll back whole). seq_t_cap > 0: nor graphs
// with more tokens of one sequence ([TAG_FN_L13_BRIDGE_SEQS]: the ring is per sequence, so 2 streams of 3 tokens fit)
// [TAG_FN_L22_TBO] n_chains >= 2 asks for a second channel per bridged layer (two-chain decode, D5). The bridge also makes
// it by itself when LLAMA_FN_L22_TBO is on (qwen4exp, spin waits) and two sequences of seq_t_cap tokens fit max_t_cap
llama_moe_bridge * llama_moe_bridge_create(const llama_model & model, int n_threads, int max_t_cap = 0, int seq_t_cap = 0,
        int n_chains = 1);
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
// [TAG_FN_L22_DMASHARE] R-REV-12: P14's share gave up (no pin in this run): later graphs fetch nothing (owner thread, the
// bridge's graph synced, no job running)
void               llama_moe_bridge_l22_dma_off(const llama_moe_bridge * br);

// graph side. active: graphs built now may use it (not paused after an error, not disabled)
bool llama_moe_bridge_active(const llama_moe_bridge * br);
int  llama_moe_bridge_max_t (const llama_moe_bridge * br);
// [TAG_FN_L13_BRIDGE_SEQS] a graph of n_tokens whose largest sequence has n_seq_tokens of them may take the bridge
bool llama_moe_bridge_fits  (const llama_moe_bridge * br, int n_tokens, int n_seq_tokens);
// [TAG_FN_L13_BRIDGE_RING] the ring rolls back a ubatch whose largest sequence has n_seq_tokens tokens
bool llama_moe_bridge_ring_ok(const llama_moe_bridge * br, int n_seq_tokens);
// [TAG_FN_WIDE_DECODE] LLAMA_FN_WIDE_DECODE: the widest decode ubatch the bridge takes beside llama_moe_bridge_fits (0: off);
// seq_lim: the most rows of one sequence the ring rolls back; wide_ub: this ubatch is one (fn_wide::decode_ub: 2+ sequences
// of the same rows, the ring rolls each back, every row an output), so it takes the bridge and the hot chain; wide_obs:
// the hot-set observer counts it as a decode step (=1; =2 keeps the old count)
int  llama_moe_bridge_wide_t (const llama_moe_bridge * br);
int  llama_moe_bridge_seq_lim(const llama_moe_bridge * br);
bool llama_moe_bridge_wide_ub(const llama_moe_bridge * br, int n_tokens, bool equal_seqs, int n_seqs, int n_seq_tokens,
        int n_outputs);
bool llama_moe_bridge_wide_obs(const llama_moe_bridge * br);
int  llama_moe_bridge_n_used(const llama_moe_bridge * br);
// the bridge id and channel for this layer's up_exps, false if the layer is not bridged
bool llama_moe_bridge_layer (const llama_moe_bridge * br, const ggml_tensor * up_exps, int32_t * id, int32_t * chan);
// [TAG_FN_L22_TBO] the channel sets of the bridge (nullptr: 0, else 1 or 2) and the channel of chain `chain` for this
// layer (chain 0 = llama_moe_bridge_layer; false for a chain the bridge does not have)
int  llama_moe_bridge_n_chains   (const llama_moe_bridge * br);
bool llama_moe_bridge_layer_chain(const llama_moe_bridge * br, const ggml_tensor * up_exps, int chain, int32_t * id, int32_t * chan);
// [TAG_FN_L22_X_DEFER] 0 off, 1 defer, 2 check (nullptr: 0); the defer channel of layer channel chan (-1: none); the rule
int     llama_moe_bridge_defer_mode  (const llama_moe_bridge * br);
int32_t llama_moe_bridge_defer_chan  (const llama_moe_bridge * br, int32_t chan);
void    llama_moe_bridge_defer_params(const llama_moe_bridge * br, float * w, int * r);

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

// [TAG_FN_L18_CHURN_GATE] LLAMA_FN_L18_CHURN_GATE (qwen4exp, spin executors): the hot set's upload copies wait here while
// a CPU expert job runs (the executor marks each job from its take to its completion), so the copies and their DMA run
// between jobs. gate_enable(true) turns the marking on (reference counted); gate_wait returns true at once when it is off
// or no job runs and the last one completed at least settle_us ago, false after max_us of waiting. gate_stats: the waits
// and their total time.
void llama_moe_bridge_gate_enable(bool on);
bool llama_moe_bridge_gate_wait(int max_us, int settle_us = 0);
void llama_moe_bridge_gate_stats(uint64_t * n_waits, uint64_t * wait_us);

// [TAG_FN_L16_PLEGATE] LLAMA_FN_L16_PLE_GATE (qwen4exp, spin waits): the bridge has one more channel that no layer posts
// to. A bridged decode graph does not wait before its launch for the PLE rows of its tokens: it waits for them on the
// device right before the PLE layer (ggml_moe_host_wait on this channel), and the PLE prefetch thread writes them there.
// A late fill times out like a late job (zeros, the bridge error, the ubatch is computed again without the bridge).
// LLAMA_FN_L16_PLE_GATE_SKIP=N (test): every Nth fill is never answered, so its wait times out.
// false: no such channel (or another row width than n_embd)
bool     llama_moe_bridge_ple_chan  (const llama_moe_bridge * br, int64_t n_embd, int32_t * id, int32_t * chan);
// owner thread, the set_input of a gated graph: the channel's next ticket (never 0). Every ticket must reach
// llama_moe_bridge_ple_fill: the bridge is not freed while a ticket is open
uint32_t llama_moe_bridge_ple_ticket(const llama_moe_bridge * br);
// any thread, no CUDA call: fill(out, ud) writes the ticket's rows (f32 [n_embd, n_tokens] in mapped memory) and returns
// false on a failure (zeros and the bridge error then); then the device wait for the ticket may go on
void     llama_moe_bridge_ple_fill  (const llama_moe_bridge * br, uint32_t ticket, int32_t n_tokens,
                                     bool (*fill)(float * out, void * ud), void * ud);

// [TAG_R38_E2] LLAMA_R38_FP4S (default 1): MoE layers with per-expert output scales (ModelOpt NVFP4 checkpoints,
// blk.N.ffn_*_exps.scale) take the Flash-Next speed paths (hot set, host bridge, DMA bank) with the scales applied there;
// 0 = v0.28.1 (those layers run the plain mul_mat_id + ggml_mul path only, the scale tensors follow the -ot/--n-cpu-moe rules)
bool llama_r38_fp4s();
