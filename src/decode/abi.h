#pragma once

// Descriptors and runtime classes shared by the megakernel runtime (src/decode/runtime.cu)
// and its Python bindings (src/bindings/).
//
// MUST COMPILE WITHOUT NVCC. The binding translation units include this header
// and are built by a plain C++ compiler, so it may not pull in CUDA headers,
// ThunderKittens or bf16. Device pointers are carried as uint64_t to keep it
// free of any CUDA or torch type.
//
// NmcLaunchDesc IS LAYOUT-LOCKED. It is the argument type of
// `mk_nmc_decode_launch_jit`, which lives in a separately compiled .so that
// decode/runtime.cu resolves with dlsym at runtime (src/jit.cpp, JitNmcFn in decode/runtime.cu). Across
// that boundary its field order and types are the contract: appending is safe,
// inserting and reordering are not. Several fields are marked "appended so
// existing offsets stay" for this reason.
//
// The other descriptors are consumed only by decode/runtime.cu and the binding TUs, which
// are all built together from this header, so they hold ordinary C++ members.
// To reference another object from one, use a smart pointer to its class, as
// kv_handle does.
//
// ADDING A FIELD requires an RW(<field>) entry in src/bindings/launch.cpp, and
// for NmcLaunchDesc a value in decode_schedule._build_launch_desc. The first omission
// fails test_launch_bindings.py, which parses this header and compares it
// against the bindings; the second fails at descriptor construction.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kv/cache.h"

namespace mk {

class JitKernel;  // declared below; the descriptors hold shared_ptrs to it.

// LAYOUT-LOCKED -- see the header comment. This crosses a dlsym boundary.
struct NmcLaunchDesc {
    int32_t num_sms;
    uint64_t stream_u64;
    uint64_t inst_buf;
    uint64_t num_inst_per_sm;
    int32_t max_inst;

    uint64_t W_upgate, W_down, W_qkv, W_oproj, W_lmhead;
    uint64_t W_router, W_moe_upgate, W_moe_down;
    uint64_t K_pool, V_pool;
    int32_t num_phys_pages;
    uint64_t page_table;
    uint64_t cache_seqlens;
    uint64_t row_active;
    int32_t max_pages_per_seq;
    int32_t page_block_size;

    uint64_t cos_table, sin_table;
    uint64_t k_cache, v_cache;

    uint64_t upgate_scratch, silu_out, q_out, o_proj_in, x_resid;
    uint64_t o_partial, lse_partial, lm_logits, rmsnorm_gamma;
    uint64_t x_raw, x_attn, x_ffn;
    uint64_t router_logits, moe_x, moe_hidden, moe_down_out;
    uint64_t topk_experts, topk_scores, topk_local_slots, route_row_for_token;
    uint64_t routed_token_ids, routed_scores, expert_counts, expert_offsets;
    int32_t max_routed;
    uint64_t moe_up_task_words, moe_down_task_words;
    uint64_t moe_up_task_count, moe_down_task_count;
    uint64_t moe_up_task_head, moe_down_task_head;
    int32_t max_moe_tasks;

    uint64_t bar_upgate, bar_silu, bar_ffn_down, bar_qkv, bar_combine;
    uint64_t bar_attn, bar_oproj, bar_layer;
    uint64_t bar_router, bar_topk, bar_route, bar_gather, bar_moe_upgate, bar_moe_down;
    uint64_t prof_buf;

    int32_t num_layers, BS, D, Dff, Hq, Hkv, head_dim, num_splits;
    int32_t timing;
    uint64_t attn_queue_words;
    uint64_t attn_queue_heads;
    int32_t attn_queue_len;
    // Per-row split counts for the drained full-attention kind. Device
    // int32[BS]; ATTN_COMBINE reads it when NUM_SPLITS is the dynamic sentinel.
    // Zero (null) when drain is off. Appended after attn_queue_len so the
    // original field offsets are unchanged.
    uint64_t attn_num_splits;
};

// One context bucket: the instruction stream and kernel to use while the
// context length is at or below bucket_upper.
//
// The variant carries claimer counts only. The attention work queue lives in
// the launch descriptor (attn_queue_words / attn_num_splits), refreshed each
// step from live cache_seqlens, and ATTN_DRAIN / ATTN_COMBINE read its length
// through the dynamic sentinel.
struct NmcRuntimeScheduleVariant {
    uint64_t inst_buf;
    uint64_t num_inst_per_sm;
    std::shared_ptr<JitKernel> jit_handle;
    int32_t max_inst;
    int32_t bucket_upper;
};

// One device region the runtime zeroes before each decode step (scratch and
// barriers). Kept POD: the runtime uploads a packed array of these to the
// device, where a kernel reads it.
struct NmcZeroRegion {
    uint64_t ptr;    // device address; may be 0 only when bytes is 0
    uint64_t bytes;
};

using NmcRuntimeTokenCallback = int (*)(
    void* context,
    const long long* sampled_tokens,
    const int32_t* emitted_rows,
    int32_t batch_size);

// Input and output for one whole lockstep decode run; see runtime_generate.
struct NmcRuntimeGenerateDesc {
    NmcLaunchDesc launch;
    uint64_t w_ln0;          // device, bf16 [D] (input_layernorm.weight[0]).
    uint64_t generated_ids;  // device int64 [BS, max_new], column 0 filled.
    std::vector<NmcZeroRegion> zero_regions;
    std::vector<int64_t> eos_token_ids;  // empty disables EOS stopping.
    uint64_t generated_lengths;  // optional device int32 [BS].
    uint64_t finish_reasons;     // optional device int32 [BS]; 0=none 1=eos 2=length.
    // ── Outputs, written by runtime_generate ────────────────────────────────
    int32_t executed_steps;
    // Per-step kernel milliseconds, sized by the runtime to the steps it ran.
    // Left empty when `timing` is 0. Entries before `warmup` stay 0.
    std::vector<float> timing_ms;
    // ────────────────────────────────────────────────────────────────────────
    int32_t max_new;
    int32_t start_pos;
    int32_t warmup;
    int32_t vocab_size;
    int32_t max_seq_len;
    std::shared_ptr<JitKernel> jit_handle;
    // Context buckets, ascending by bucket_upper. Empty falls back to
    // jit_handle.
    std::vector<NmcRuntimeScheduleVariant> schedule_variants;
    // Shared ownership: the runtime holds these KV blocks for the whole call.
    std::shared_ptr<kv_pool::KvHandle> kv_handle;
    float rms_norm_eps;
    // Zero selects greedy argmax. Positive values select Gumbel-max sampling
    // from logits / temperature; top-p filtering is intentionally unsupported.
    float temperature;
    // Request-scoped seed supplied by Python. The sampling kernel combines it
    // with the decode step, batch row, and vocabulary index in a stateless
    // Philox counter, so sampling needs no mutable device RNG state.
    uint64_t sampling_seed;
    int32_t timing;
    // Callback and its context; see the note above NmcDecodeServiceDesc.
    uint64_t token_callback;          // optional NmcRuntimeTokenCallback.
    uint64_t token_callback_context;  // opaque host pointer passed through.
    // Per-row prompt lengths, for ragged batches. Read on the CPU to build the
    // KV position vector and pick the schedule bucket. Empty means every row
    // starts at `start_pos`. When set, `start_pos` must hold the maximum, which
    // keeps the horizon check and caller-side sizing conservative.
    std::vector<int32_t> start_pos_per_row;
    // Per-step attention-drain queue rebuild. When attn_drain != 0 the loop
    // refreshes launch.attn_queue_words / attn_num_splits from live row lengths
    // using max_attn_splits / min_attn_chunk; page_block, Hkv, BS and num_sms
    // come from launch.
    int32_t attn_drain;
    int32_t max_attn_splits;
    int32_t min_attn_chunk;
};

// Geometry and buffers for the continuous-batching decode service.
//
// WHAT THE CALLER MUST KEEP ALIVE, for as long as the service runs or until its
// next set_geometry call:
//   - The device addresses (w_ln0, generated_ids, d_gen_col, d_temperature,
//     d_seed) and those inside launch and schedule_variants. They point at
//     torch tensors that Python owns.
//   - The token-callback object behind token_callback_context. session holds
//     it as NmcDecodeSession.callback.
// Everything else is owned by the descriptor and copied into the service.
//
// Converting token_callback to an owning std::function would capture a Python
// object; the service destroys its descriptor copy from close() with the GIL
// released, where decrefing crashes.
struct NmcDecodeServiceDesc {
    NmcLaunchDesc launch;         // template for this geometry.
    uint64_t w_ln0;               // device, bf16 [D].
    uint64_t generated_ids;       // device int64 [BS, max_new].
    std::vector<NmcZeroRegion> zero_regions;
    std::vector<int64_t> eos_token_ids;  // empty disables EOS stopping.
    int32_t max_new;              // width of generated_ids per row.
    int32_t vocab_size;
    int32_t max_seq_len;
    // Shared ownership. A geometry switch swaps in the rebound handle; the
    // previous one is released when the last descriptor holding it drops,
    // giving set_geometry the commit point it needs.
    std::shared_ptr<kv_pool::KvHandle> kv_handle;
    float rms_norm_eps;
    // Per-row control device arrays (length BS). The service owns authoritative
    // host copies and uploads to these each step; kernels read the device side.
    uint64_t d_gen_col;           // int32 [BS].
    uint64_t d_temperature;       // float [BS].
    uint64_t d_seed;              // uint64 [BS].
    // Context buckets at the fixed BS, ascending by bucket_upper. Empty falls
    // back to jit_handle.
    std::vector<NmcRuntimeScheduleVariant> schedule_variants;
    std::shared_ptr<JitKernel> jit_handle;
    uint64_t token_callback;          // optional NmcRuntimeTokenCallback.
    uint64_t token_callback_context;  // opaque host pointer passed through.
    // Per-step attention-drain queue rebuild (same contract as
    // NmcRuntimeGenerateDesc).
    int32_t attn_drain;
    int32_t max_attn_splits;
    int32_t min_attn_chunk;
};

// ─────────────────────────────────────────────────────────────────────────────
// The runtime implemented by src/decode/runtime.cu.
//
// Everything below reports failure by throwing MkError. DecodeService::run is
// the single exception; see its comment.

// Thrown by everything below. src/bindings/launch.cpp surfaces it in Python as
// `MkAbiError`, a RuntimeError subclass.
class MkError : public std::runtime_error {
 public:
    using std::runtime_error::runtime_error;
};

// One compiled megakernel variant: the cached .so, its dlopen handle and its
// resolved launch symbol.
//
// Held by shared_ptr because a schedule variant, a generate descriptor and a
// running decode service can all name the same kernel, and the last of them to
// finish is the one that may unload it.
class JitKernel {
 public:
    // Compiles, or fetches from the little_jit cache, the megakernel for one
    // batch size. Blocks for as long as nvcc takes; do not call under the GIL.
    static std::shared_ptr<JitKernel> compile(int32_t bs, bool enable_profiler,
                                              const std::string& repo_root,
                                              const std::string& config_json);
    ~JitKernel();

    JitKernel(const JitKernel&) = delete;
    JitKernel& operator=(const JitKernel&) = delete;

    // Launches one decode step and returns its CUDA-event milliseconds. The
    // launch is asynchronous with respect to the device.
    float decode_launch(const NmcLaunchDesc& desc);
    // Filesystem path of the compiled .so.
    const std::string& artifact_path() const;

    struct Impl;

 private:
    explicit JitKernel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> p_;
};

// A device profiler buffer. Owns its device allocation.
class Profiler {
 public:
    Profiler(int32_t num_sms, int32_t max_events);
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    // Device address to hand to NmcLaunchDesc::prof_buf.
    uint64_t device_ptr() const;
    void init();  // reset the event cursors
    void export_to(const std::string& filename);

 private:
    struct sm_profiler_buffer* buffer_ = nullptr;
};

// The long-lived decode service (continuous batching), driven by
// src/serving/session.py.
//
// THREADING: one thread calls run() and blocks in it; other threads drive the
// loop through the signalling methods. Bulk batch-composition mutation is legal
// only while the loop is parked.
class DecodeService {
 public:
    // Copies the descriptor, sharing ownership of its kv_handle and jit
    // handles. See NmcDecodeServiceDesc for what the caller must keep alive.
    explicit DecodeService(const NmcDecodeServiceDesc& desc);
    ~DecodeService();

    DecodeService(const DecodeService&) = delete;
    DecodeService& operator=(const DecodeService&) = delete;

    // Runs the decode loop until stopped. Blocks for the service's lifetime.
    //
    // Returns a status code so the caller can tell an ordinary failure apart
    // from the watchdog-expired code, which terminates the process instead of
    // unwinding. last_error() carries the message.
    int32_t run();

    void signal_pause(bool paused);  // asynchronous; pair with wait_paused
    void signal_stop();
    // Blocks until the loop parks, or the timeout elapses. False means it did
    // not park.
    bool wait_paused(int32_t timeout_ms);
    void notify();  // wake the loop after changing slot state

    // Populates one batch row. The service must be paused.
    void set_slot(int32_t row, bool active, int32_t start_pos, int32_t gen_col,
                  float temperature, uint64_t seed);
    // Writes batch_size() entries into each non-null output array.
    void get_state(int32_t* out_active, int32_t* out_gen_col,
                   int32_t* out_finish);
    // The service's current batch size. Size the get_state buffers with this:
    // a caller-side batch that went stale across a geometry switch overruns
    // them.
    int32_t batch_size();

    // Switches decode geometry (session batch size) mid-flight. The caller MUST
    // hold the service paused and re-populate slot state afterwards.
    void set_geometry(const NmcDecodeServiceDesc& desc);

    // Non-zero when the loop parked because the KV pool cannot satisfy the next
    // step. The Python scheduler polls this to trigger preemption.
    int32_t kv_pressure();
    void resume_from_pressure();

    // Stops the loop and drops this object's reference to it. In-flight calls
    // hold their own reference to the impl, so close() cannot free the service
    // underneath them.
    void close();

    class Impl;

 private:
    std::shared_ptr<Impl> p_;
};

// ── Free functions: no object is involved ───────────────────────────────────

// Message from the most recent DecodeService::run failure on this thread.
// Everything else reports through MkError.
const char* last_error();

// Widens host-side watchdog diagnostics. MUST be called before creating a
// decode service: geometry setup only allocates the snapshot buffers when debug
// is already on, so flipping it later leaves them missing.
void set_debug(bool enabled);

// Per-row split counts from the native attention-drain policy. This is a test
// hook: it exposes the policy that actually runs so a test can compare it with
// decode_schedule._attn_drain_splits_by_row. The decode path reaches the same
// policy internally.
std::vector<int32_t> attn_drain_splits(const std::vector<int32_t>& cache_seqlens,
                                       int32_t hkv, int32_t num_sms,
                                       int32_t page_block,
                                       int32_t max_attn_splits,
                                       int32_t min_attn_chunk,
                                       int32_t oversub_k);

// Runs the whole one-shot lockstep decode loop. Blocks for the entire
// generation; the device buffers the descriptor points at must outlive the
// call. Writes executed_steps and timing_ms back into `desc`.
void runtime_generate(NmcRuntimeGenerateDesc& desc);

}  // namespace mk
