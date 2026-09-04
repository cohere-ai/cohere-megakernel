// decode/megakernel.cuh -- NMC decode megakernel.
//
// Defines the device instruction ABI, JIT-specialized tiling configs,
// persistent controller/worker pipeline, dense and MoE GEMMs, paged attention,
// routing, and residual RMSNorm operations used by the release runtime.

#pragma once

#include "decode/gemm-n8-wgmma.cuh"
#include "kittens.cuh"
#include "sm_profiler.h"
#include <cuda/barrier>
#include <cstdint>
#include <type_traits>

namespace mk {

using namespace kittens;
using bf16 = __nv_bfloat16;

// ── instruction stream ───────────────────────────────────────────────────────
// Stable integer values: the host-side encoder writes these directly into the
// int32 instruction buffer.
// Compile-time roles discard unreachable op paths, letting ptxas size registers
// per warpgroup and honor `setmaxnreg`.
enum class NmcRole : int {
    PRODUCER = 0,   // WG0 warp 1 — TMA-loads, sem inits, cross-SM input bar waits
    STORER   = 1,   // WG0 warp 2 — TMA-stores (gemm_op only) + cross-SM output bar arrives
    CONSUMER = 2,   // WG1 (CWG_IDX=0) and WG2 (CWG_IDX=1, when NUM_CWG==2)
    IDLE     = 3,   // WG0 warp 3 (or the optional second MoE producer)
};

enum class NmcOpcode : int {
    NOP            = 0,
    FFN_DOWN       = 1,
    QKV_PROJ       = 2,
    ATTN_DECODE    = 3,
    ATTN_COMBINE   = 4,
    O_PROJ         = 5,
    LM_HEAD        = 6,
    FFN_UPGATE_ACT = 7,
    ATTN_DRAIN     = 8,
    ROUTER_GEMM    = 9,
    ROUTER_TOPK    = 10,
    ROUTE_FINALIZE = 11,
    MOE_GATHER     = 12,
    MOE_UPGATE_ACT_DRAIN = 13,
    MOE_DOWN_DRAIN = 14,
    MOE_COMBINE    = 15,
    ADD_RMSNORM    = 16,
    // Ablation-only: host inserts one per SM between dependency waves so the
    // grid cannot overlap wave N+1 with unfinished wave-N SMs. Not used
    // by the production schedule.
    GRID_SYNC      = 17,
};

// One profiler stream is shared by the whole CTA. Thread 0 initializes the
// group-0 context, while producer lane 0 is the only thread that mutates the
// range state. This deliberately avoids per-warp tracks in the release trace.
struct NmcBlockProfiler {
    SmProfilerCtx ctx{};
    uint32_t handle{SM_PROFILER_INVALID_HANDLE};
    uint32_t event_no{};
    bool scope_open{};
};

struct NmcDisabledBlockProfiler {};

template <class Cfg>
using NmcBlockProfilerStorage = std::conditional_t<
    Cfg::ENABLE_PROFILER, NmcBlockProfiler, NmcDisabledBlockProfiler>;

template <class Cfg>
__device__ __forceinline__ SmProfilerCtx prof_ctx(uint64_t* prof_buf) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        if constexpr (Cfg::PROFILE_BY_SM) {
            return sm_profiler_init_ctx_by_smid(prof_buf);
        } else {
            return sm_profiler_init_ctx(prof_buf);
        }
    } else {
        return {};
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_init(
    Profiler& profiler, uint64_t* prof_buf) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        profiler.ctx = prof_ctx<Cfg>(prof_buf);
        profiler.handle = SM_PROFILER_INVALID_HANDLE;
        profiler.event_no = static_cast<uint32_t>(NmcOpcode::NOP);
        profiler.scope_open = false;
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_prepare(
    Profiler& profiler, NmcOpcode opcode) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        profiler.handle = SM_PROFILER_INVALID_HANDLE;
        profiler.event_no = static_cast<uint32_t>(opcode);
        profiler.scope_open = false;
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_begin(Profiler& profiler) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (!profiler.scope_open) {
            profiler.scope_open = true;
            profiler.handle = sm_profiler_start(profiler.ctx, profiler.event_no);
        }
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_pause(Profiler& profiler) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (profiler.scope_open && profiler.handle != SM_PROFILER_INVALID_HANDLE) {
            sm_profiler_end(profiler.ctx, profiler.handle);
        }
        profiler.handle = SM_PROFILER_INVALID_HANDLE;
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_resume(Profiler& profiler) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (profiler.scope_open) {
            profiler.handle = sm_profiler_start(profiler.ctx, profiler.event_no);
        }
    }
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void block_profiler_finish(Profiler& profiler) {
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (profiler.scope_open && profiler.handle != SM_PROFILER_INVALID_HANDLE) {
            sm_profiler_end(profiler.ctx, profiler.handle);
        }
        profiler.handle = SM_PROFILER_INVALID_HANDLE;
        profiler.scope_open = false;
    }
}

// Embedding lookup and the initial RMSNorm are deliberately outside this
// megakernel. The direct Python launcher or native decode service seeds
// `x_raw`/`x_resid` before each persistent-kernel launch.

// One instruction = NMC_INSTRUCTION_WIDTH int32 fields.  Field 0 is the opcode;
// remaining fields are op-specific (layer_idx, tile_m, tile_n, window, …).
// The fixed width of 32 leaves room for encoder-side ABI extensions.
constexpr int NMC_INSTRUCTION_WIDTH = 32;

struct NmcInstruction {
    int data[NMC_INSTRUCTION_WIDTH];
    __device__ __forceinline__ NmcOpcode opcode() const {
        return static_cast<NmcOpcode>(data[0]);
    }
};

// Common GEMM instruction layout (FFN_UPGATE_ACT / FFN_DOWN / O_PROJ / LM_HEAD).
// Fields beyond [4] are op-specific; QKV uses a separate layout.
namespace gemm_field {
    constexpr int OPCODE          = 0;
    constexpr int LAYER           = 1;
    constexpr int TILE_M          = 2;
    constexpr int TILE_N          = 3;
    constexpr int SPLIT           = 4;
    constexpr int K_TILES         = 5;   // total K-tiles in K dim (K / BK)
    constexpr int WAIT_BAR_IDX    = 6;   // input cross-SM bar slot
    constexpr int WAIT_TARGET     = 7;   // threshold for wait_bar_idx
    constexpr int PRODUCE_BAR_IDX = 8;   // output cross-SM bar slot
    constexpr int NUM_TILES       = 9;   // number of packed output tile coords
    constexpr int TILE_IDS        = 10;  // packed output tile coords: (tile_m << 16) | tile_n
}

// ---- Packed MoE task record (dynamic drain path) ---------------------------
// route_finalize emits one int4 per (expert, tile_m, split), replacing several
// strided stores in this store-bound path. `expert` is derived from
// layer_expert % NMC_NUM_EXPERTS and is not stored.
//
// All task-buffer access must use these helpers. MOE_TASK_RECORD_INTS must stay
// synchronized with decode/schedule.py.
struct alignas(16) MoeTaskRecord {
    int layer_expert;   // layer * NMC_NUM_EXPERTS + expert
    int split;          // split-k index
    int wait_target;    // down tasks: up_wait_target; up tasks: 0 (unused)
    int tile_ids;       // (tile_m << 16) | tile_n
};
static_assert(sizeof(MoeTaskRecord) == 16, "MoeTaskRecord must pack into one int4");
constexpr int MOE_TASK_RECORD_INTS = sizeof(MoeTaskRecord) / sizeof(int);  // 4

__device__ __forceinline__ void store_moe_task(
    int* task_words, int layer, int max_moe_tasks, int task, const MoeTaskRecord& rec) {
    int4* slot = reinterpret_cast<int4*>(
        task_words + ((size_t)layer * (size_t)max_moe_tasks + (size_t)task) * MOE_TASK_RECORD_INTS);
    *slot = make_int4(rec.layer_expert, rec.split, rec.wait_target, rec.tile_ids);
}

__device__ __forceinline__ MoeTaskRecord load_moe_task(
    const int* task_words, int layer, int max_moe_tasks, int task) {
    const int4 v = *reinterpret_cast<const int4*>(
        task_words + ((size_t)layer * (size_t)max_moe_tasks + (size_t)task) * MOE_TASK_RECORD_INTS);
    return MoeTaskRecord{v.x, v.y, v.z, v.w};
}

// QKV_PROJ instruction layout. Each instruction handles one (gemm_idx,
// tile_m, tile_n) chunk: gemm_idx ∈ {0=Q, 1=K, 2=V}. Q and K may apply
// RoPE in-register; V never does.
namespace qkv_field {
    constexpr int OPCODE          = 0;
    constexpr int LAYER           = 1;
    constexpr int TILE_M          = 2;
    constexpr int TILE_N          = 3;   // head_idx for Q; kv_head_idx for K/V
    constexpr int GEMM_IDX        = 4;   // 0=Q, 1=K, 2=V
    constexpr int NEEDS_ROPE      = 5;
    constexpr int K_TILES         = 6;
    constexpr int WAIT_BAR_IDX    = 7;
    constexpr int WAIT_TARGET     = 8;
    constexpr int PRODUCE_BAR_IDX = 9;
}

// ATTN_DECODE instruction layout. One CTA processes one (batch, kv_head,
// split) slice; encoder fans out across SMs.
namespace attn_field {
    constexpr int OPCODE          = 0;
    constexpr int LAYER           = 1;
    constexpr int BATCH_IDX       = 2;
    constexpr int KV_HEAD_IDX     = 3;
    constexpr int SPLIT_IDX       = 4;
    constexpr int NUM_SPLITS      = 5;
    constexpr int WINDOW_SIZE     = 6;   // 0 = full attention
    constexpr int WAIT_TARGET     = 8;
    constexpr int NUM_TILES       = 10;  // number of packed (batch, kv_head) coords
    constexpr int TILE_IDS        = 11;  // packed coords: (batch_idx << 16) | kv_head
}

// ATTN_DRAIN instruction layout. Each CTA drains a layer-local attention
// queue. Queue entries are ordinary ATTN_DECODE instruction words.
namespace attn_drain_field {
    constexpr int OPCODE       = 0;
    constexpr int LAYER        = 1;
    constexpr int QUEUE_OFFSET = 2;
    constexpr int QUEUE_LEN    = 3;
}

// ATTN_COMBINE instruction layout. One CTA per (batch, kv_head).
namespace combine_field {
    constexpr int OPCODE          = 0;
    constexpr int LAYER           = 1;
    constexpr int BATCH_IDX       = 2;
    constexpr int KV_HEAD_IDX     = 3;
    constexpr int NUM_SPLITS      = 4;
    constexpr int WAIT_BAR_IDX    = 5;
    constexpr int WAIT_TARGET     = 6;
    constexpr int PRODUCE_BAR_IDX = 7;
}

// Zero selects host-written queue length/per-row split counts. Static schedules
// bake positive values, and the dynamic host policy floors splits at two.
constexpr int ATTN_DYNAMIC_SENTINEL = 0;

namespace router_topk_field {
    constexpr int OPCODE = 0;
    constexpr int LAYER = 1;
    constexpr int ROW = 2;
}

namespace route_finalize_field {
    constexpr int OPCODE = 0;
    constexpr int LAYER = 1;
}

namespace moe_gather_field {
    constexpr int OPCODE = 0;
    constexpr int LAYER = 1;
    constexpr int ROW = 2;
}

namespace moe_combine_field {
    constexpr int OPCODE = 0;
    constexpr int LAYER = 1;
    constexpr int ROW = 2;
}

namespace rmsnorm_field {
    constexpr int OPCODE = 0;
    constexpr int LAYER = 1;
    constexpr int ROW = 2;
    // >0: wait bar_ffn_down[layer] >= target (dense FFN / baseline MoE combine).
    // -1: MOE_COMBINE_ATOMIC_TMA sentinel -- wait bar_route then this row's
    //     topk fine-grained bar_moe_down blocks (no MOE_COMBINE arrivals).
    //  0: skip FFN wait.
    constexpr int WAIT_FFN_TARGET = 3;
    constexpr int WAIT_OPROJ_TARGET = 4;
    constexpr int PRODUCE_BAR_IDX = 5;
}

constexpr int NMC_NUM_EXPERTS = 128;
constexpr int NMC_TOPK = 8;
constexpr int NMC_EXPERT_DFF = 768;
constexpr int NMC_MOE_SCHED_RING = 2;

// Stream once-read weights/KV through L2 with EVICT_FIRST and retain reused
// activations with EVICT_LAST. These are performance-only hints.
#ifndef NMC_MOE_L2_HINT
#define NMC_MOE_L2_HINT 1
#endif

// Fine-grained MoE down -> combine dependency. When defined, MoE down tasks
// arrive on the per-(layer, expert, down-row-block) bar_moe_down slot.
// #define NMC_MOE_FINE_GRAINED_DOWN_COMBINE

// debug flags, turn off for release
#ifndef NMC_TRAP_ON_ERROR
#define NMC_TRAP_ON_ERROR 0
#endif

// Fixed attention ABI for this release kernel.
constexpr int HEAD_DIM         = 128;
constexpr int ATTN_BLOCK_KV    = 64;
constexpr int Q_TILE_H         = 16;
constexpr int ATTN_NUM_STAGES  = 3;

// Kernel-level storage makes all role-specialized loops share semaphore
// addresses; per-instantiation copies would deadlock producers and consumers.
constexpr int MAX_GEMM_STAGES = 16;   // upper bound across all per-op cfgs
template <class Cfg>
struct NmcOpSmem {
    [[no_unique_address]] NmcBlockProfilerStorage<Cfg> profiler;
    kittens::semaphore gemm_data_bar [MAX_GEMM_STAGES];
    kittens::semaphore gemm_empty_bar[MAX_GEMM_STAGES];
    kittens::semaphore gemm_store_done_bar;
    kittens::semaphore attn_k_bar    [ATTN_NUM_STAGES];
    kittens::semaphore attn_v_bar    [ATTN_NUM_STAGES];
    kittens::semaphore attn_empty_bar[ATTN_NUM_STAGES];
    uint32_t row_active_bits; // one bit per batch row; BS is capped at 8
    int attn_claim_active;
    NmcInstruction attn_claim_inst;
    // MoE drain uses this tiny metadata ring to publish queue-claimed GEMM
    // tasks from the producer warp to consumers/storer without a CTA-wide
    // rendezvous at each tile. The GEMM data pipeline remains in
    // gemm_data_bar/gemm_empty_bar; this only carries per-tile instruction
    // words and an epoch marker.
    int moe_sched_active[NMC_MOE_SCHED_RING];
    int moe_sched_epoch[NMC_MOE_SCHED_RING];
    int moe_sched_batch_base;
    NmcInstruction moe_sched_inst[NMC_MOE_SCHED_RING];
};

// ── globals ──────────────────────────────────────────────────────────────────
// All weight / TMA-touchable buffers use TK gl<>; layer is the leading TMA
// coordinate in the stacked descriptors.
template <class Cfg>
struct NmcGlobals {
    using upgate_w_tile = st_bf<Cfg::UpGate::BN, Cfg::UpGate::BK>;
    using down_w_tile   = st_bf<Cfg::Down  ::BN, Cfg::Down  ::BK>;
    using qkv_w_tile    = st_bf<Cfg::QKV   ::BN, Cfg::QKV   ::BK>;
    using oproj_w_tile  = st_bf<Cfg::OProj ::BN, Cfg::OProj ::BK>;
    using lmhead_w_tile = st_bf<Cfg::LMHead::BN, Cfg::LMHead::BK>;
    using router_w_tile = st_bf<Cfg::Router::BN, Cfg::Router::BK>;
    using moe_upgate_w_tile = st_bf<Cfg::MoeUpGate::BN, Cfg::MoeUpGate::BK>;
    using moe_down_w_tile = st_bf<Cfg::MoeDown::BN, Cfg::MoeDown::BK>;
    // Activation gls play two roles: written as a non-swizzled y_tile by one
    // op, then read as a default-swizzled x_tile by the next. The gl must
    // register descriptors for both tile types.
    using act_y_tile    = st_bf<16, 64, false>;  // store side
    using act_x_tile    = st_bf<16, 64>;         // load side

    // instructions — per-SM lists (persistent kernel: grid = #SMs, each CTA
    // processes its own slice).
    int*       inst_buf;          // [num_sms, max_inst, NMC_INSTRUCTION_WIDTH]
    const int* num_inst_per_sm;   // [num_sms]
    int        max_inst;          // pitch of inst_buf along the inst dim

    // weights (3D-stacked by layer; lmhead has L=1)
    // 4D: (1, num_layers, N, K) — leading singleton; layer is depth coord.
    gl<bf16, 1, -1, -1, -1, upgate_w_tile> W_upgate;
    gl<bf16, 1, -1, -1, -1, down_w_tile  > W_down;
    gl<bf16, 1, -1, -1, -1, qkv_w_tile   > W_qkv;
    gl<bf16, 1, -1, -1, -1, oproj_w_tile > W_oproj;
    gl<bf16, 1, 1,  -1, -1, lmhead_w_tile> W_lmhead;  // L=1
    gl<bf16, 1, -1, -1, -1, router_w_tile> W_router;
    gl<bf16, 1, -1, -1, -1, moe_upgate_w_tile> W_moe_upgate;
    gl<bf16, 1, -1, -1, -1, moe_down_w_tile> W_moe_down;

    // paged KV cache
    using k_tile = st_bf<ATTN_BLOCK_KV, HEAD_DIM>;
    using v_tile = st_bf<ATTN_BLOCK_KV, HEAD_DIM>;
    gl<bf16, -1, -1, -1, -1, k_tile> K;   // (num_phys_pages, page_block, Hkv, D_h)
    gl<bf16, -1, -1, -1, -1, v_tile> V;
    int* page_table;       // (L, BS, max_pages_per_seq)
    int* cache_seqlens;    // (BS,)
    const int* row_active; // optional (BS,), nullptr means all rows active
    int  max_pages_per_seq;
    int  page_block_size;

    // RoPE tables (precomputed cos/sin for each (pos, freq)). Each table is
    // (max_seq_len, HEAD_DIM/2) f32, indexed [pos * (HEAD_DIM/2) + freq].
    const float* cos_table;
    const float* sin_table;
    // Raw KV cache pointers (used by QKV scatter; same memory as K/V gls).
    bf16* k_cache;
    bf16* v_cache;

    // single-layer activations / scratch (overwritten each layer)
    // 4D: (1, 1, BS, Dim) — single-layer activations, overwritten each layer.
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> upgate_scratch;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> silu_out;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> q_out;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> o_proj_in;
    // x_resid/x_attn/x_ffn live in HBM as (BS, D) bf16; wrap the GEMM
    // outputs in 4D gls so FFN_DOWN / O_PROJ can use TMA stores.
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> x_resid;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> x_attn_gl;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> x_ffn_gl;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> router_logits;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> moe_x;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> moe_hidden;
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> moe_down_out;
    bf16*  o_partial;         // (num_splits, BS, Hkv, hr, D_h)
    float* lse_partial;       // (num_splits, BS, Hkv, hr)
    // (BS, VOCAB) bf16. LM_HEAD's TMA store uses act_y_tile so this gl
    // shares the activation tile dims; col_tile selects the BN-sized
    // vocab slice and the gl bounds-clip the partial last tile.
    gl<bf16, 1, 1, -1, -1, act_y_tile, act_x_tile> lm_logits;
    bf16* rmsnorm_gamma;      // (L, D), NMC RMSNorm scale
    bf16*  x_raw;             // running residual sum (read+write)
    bf16*  x_attn;            // OProj output to add (read; zeroed by RMSNorm)
    bf16*  x_ffn;             // FFN DOWN output to add (read; zeroed by RMSNorm)
    int* topk_experts;        // (L, BS, NMC_TOPK)
    bf16* topk_scores;        // sigmoid(router) for selected experts
    int* topk_local_slots;    // local rank inside each expert bucket
    int* route_row_for_token; // (L, BS, NMC_TOPK) -> routed row
    int* routed_token_ids;    // (L, max_routed)
    bf16* routed_scores;      // (L, max_routed)
    int* expert_counts;       // (L, NMC_NUM_EXPERTS)
    int* expert_offsets;      // (L, NMC_NUM_EXPERTS + 1)
    int max_routed;
    int* moe_up_task_words;   // (L, max_moe_tasks, MOE_TASK_RECORD_INTS)
    int* moe_down_task_words; // (L, max_moe_tasks, MOE_TASK_RECORD_INTS)
    int* moe_up_task_count;   // (L,)
    int* moe_down_task_count; // (L,)
    uint32_t* moe_up_task_head;
    uint32_t* moe_down_task_head;
    int max_moe_tasks;

    // cross-SM barriers (uint32 atomic counters)
    uint32_t* bar_upgate;
    uint32_t* bar_silu;
    uint32_t* bar_ffn_down;
    uint32_t* bar_qkv;
    uint32_t* bar_combine;
    uint32_t* bar_attn;
    uint32_t* bar_oproj;
    uint32_t* bar_layer;
    uint32_t* bar_router;
    uint32_t* bar_topk;
    uint32_t* bar_route;
    uint32_t* bar_gather;
    uint32_t* bar_moe_upgate;
    uint32_t* bar_moe_down;

    // optional SM-profiler device buffer; nullptr disables profiling.
    uint64_t* prof_buf;

    // shapes
    int num_layers, BS, D, Dff, Hq, Hkv, num_splits;

    // Host-built full-attention queue and live per-row split counts. Dynamic
    // sentinel fields select these values; sliding layers use static schedules.
    // attn_num_splits persists across steps and must not be reset as scratch.
    int* attn_queue_words;
    uint32_t* attn_queue_heads;
    int attn_queue_len;
    int* attn_num_splits;  // (BS,), nullptr when drain is off
};

// ── warp-role helpers ────────────────────────────────────────────────────────
__device__ __forceinline__ int  lane_id()    { return threadIdx.x & 31; }
__device__ __forceinline__ int  warp_id()    { return threadIdx.x >> 5; }
__device__ __forceinline__ int  wg_id()      { return warp_id() >> 2; }
__device__ __forceinline__ int  warp_in_wg() { return warp_id() & 3; }

template <class Cfg>
__device__ __forceinline__ uint32_t load_row_active_bits(const NmcGlobals<Cfg>& g) {
    if (g.row_active == nullptr) return 0xffffffffu;
    uint32_t bits = 0u;
    for (int row = 0; row < g.BS; ++row) {
        bits |= (g.row_active[row] != 0 ? 1u : 0u) << row;
    }
    return bits;
}

template <class Cfg>
__device__ __forceinline__ bool row_is_active(
    const NmcGlobals<Cfg>& g, uint32_t row_active_bits, int row) {
    if (row < 0 || row >= g.BS) return false;
    return ((row_active_bits >> row) & 1u) != 0u;
}

template <class Cfg>
__device__ __forceinline__ bool any_active_row_in_tile(
    const NmcGlobals<Cfg>& g, uint32_t row_active_bits, int row_tile, int tile_rows) {
    const int row_begin = row_tile * tile_rows;
    const int row_end = min(row_begin + tile_rows, g.BS);
    if (row_begin >= row_end) return false;
    const int width = row_end - row_begin;
    const uint32_t mask = ((1u << width) - 1u) << row_begin;
    return (row_active_bits & mask) != 0u;
}

// Worker-only counting barrier; the controller remains in its prefetch loop.
// Every worker warp must execute the same number of worker_sync calls on every
// path. Otherwise arrivals from different logical syncs can satisfy each other,
// letting warps drift onto different instructions before an unrelated deadlock.
// The NOP dispatch pads calls to preserve this invariant.
template <class Cfg>
__device__ __forceinline__ void worker_sync() {
    constexpr int NW = Cfg::NUM_THREADS - 32;
    asm volatile("bar.sync 1, %0;" :: "r"(NW) : "memory");
}

// ── TMA bulk-group store helpers ─────────────────────────────────────────────
__device__ __forceinline__ void gemm_tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}
__device__ __forceinline__ void gemm_tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;");
}
template <int Count>
__device__ __forceinline__ void gemm_tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(Count) : "memory");
}

// Non-tensor TMA reduce-add into global bf16. PTX requires .noftz for bf16/f16
// floating add (preserves subnormals). gdst/ssrc must be 16B-aligned; bytes
// must be a multiple of 16. Completion uses the same bulk-group commit/wait
// helpers as ordinary TMA stores (gemm_tma_store_arrive / gemm_tma_store_wait).
// Used by the MOE_COMBINE_ATOMIC_TMA arm to scatter-reduce scaled MoE-down
// tiles directly into x_ffn, skipping the MOE_COMBINE op.
__device__ __forceinline__ void tma_reduce_add_noftz_bf16(
    void* gdst, const void* ssrc, uint32_t bytes) {
    uint32_t ssrc_u32;
    asm volatile("{"
        ".reg .u64 smem_ptr64;\n\t"
        "cvta.to.shared.u64 smem_ptr64, %1;\n\t"
        "cvt.u32.u64 %0, smem_ptr64;\n\t"
        "}"
        : "=r"(ssrc_u32) : "l"(ssrc));
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[%0], [%1], %2;\n"
        :: "l"(gdst), "r"(ssrc_u32), "r"(bytes)
        : "memory");
}

// ── cross-SM bar helpers ─────────────────────────────────────────────────────
// Each cross-SM barrier is a single uint32 atomic counter; producers add to
// it, consumers spin until it crosses a threshold. We ignore null pointers
// so ops can be tested in isolation.
//
// Acquire (waiter): after the counter crosses the threshold, issue a
// threadfence so generic global-memory reads see prior writers' data.
__device__ __forceinline__ void wait_cross_sm(const uint32_t* bar, uint32_t target) {
    if (bar == nullptr) return;
    while (*(volatile const uint32_t*)bar < target) {
        __nanosleep(20); // prevent too much memory traffic, DONT DELETE
    }
    __threadfence();
}
template <class Cfg, class Profiler>
__device__ __forceinline__ void profiled_wait_cross_sm(
    const uint32_t* bar, uint32_t target, Profiler& profiler) {
    if (bar == nullptr) return;
    bool paused = false;
    if constexpr (Cfg::ENABLE_PROFILER) {
        // Do not manufacture a visible gap when the dependency is already
        // satisfied. Only producer lane 0 owns the shared profiler state.
        if (threadIdx.x == kittens::WARP_THREADS && profiler.scope_open &&
            *(volatile const uint32_t*)bar < target) {
            block_profiler_pause<Cfg>(profiler);
            paused = true;
        }
    }
    wait_cross_sm(bar, target);
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (threadIdx.x == kittens::WARP_THREADS && paused) {
            block_profiler_resume<Cfg>(profiler);
        }
    }
}
// Release (signaller): TMA / async-proxy stores must be visible before other
// SMs observe the counter bump.
__device__ __forceinline__ void arrive_cross_sm(uint32_t* bar, uint32_t inc = 1) {
    if (bar == nullptr) return;
    asm volatile("fence.proxy.async;\n" ::: "memory");
    __threadfence();
    atomicAdd(bar, inc);
}

__device__ __forceinline__ void wait_cross_sm_indexed(
    const uint32_t* bar, int idx, uint32_t target) {
    if (bar == nullptr) return;
    wait_cross_sm(&bar[idx], target);
}

template <class Cfg, class Profiler>
__device__ __forceinline__ void profiled_wait_cross_sm_indexed(
    const uint32_t* bar, int idx, uint32_t target,
    Profiler& profiler) {
    if (bar == nullptr) return;
    profiled_wait_cross_sm<Cfg>(&bar[idx], target, profiler);
}

template <int N>
__device__ __forceinline__ bool cross_sm_all_indexed_reached(
    const uint32_t* bar, const int (&indices)[N], uint32_t target) {
    static_assert(N > 0, "cross-SM multi-wait requires at least one counter");
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        if (*(volatile const uint32_t*)&bar[indices[i]] < target) return false;
    }
    return true;
}

template <int N>
__device__ __forceinline__ void wait_cross_sm_all_indexed(
    const uint32_t* bar, const int (&indices)[N], uint32_t target) {
    if (bar == nullptr) return;
    while (!cross_sm_all_indexed_reached(bar, indices, target)) {
        __nanosleep(20);
    }
    // One acquire fence is sufficient after every required counter is ready.
    // Calling wait_cross_sm_indexed N times would issue N redundant fences.
    __threadfence();
}

template <class Cfg, int N, class Profiler>
__device__ __forceinline__ void profiled_wait_cross_sm_all_indexed(
    const uint32_t* bar, const int (&indices)[N], uint32_t target,
    Profiler& profiler) {
    if (bar == nullptr) return;
    bool paused = false;
    if constexpr (Cfg::ENABLE_PROFILER) {
        // Match the single-counter profiler contract: dependency stalls are
        // excluded from the current opcode range, but an already-ready group
        // does not manufacture a visible trace gap.
        if (threadIdx.x == kittens::WARP_THREADS && profiler.scope_open &&
            !cross_sm_all_indexed_reached(bar, indices, target)) {
            block_profiler_pause<Cfg>(profiler);
            paused = true;
        }
    }
    wait_cross_sm_all_indexed(bar, indices, target);
    if constexpr (Cfg::ENABLE_PROFILER) {
        if (threadIdx.x == kittens::WARP_THREADS && paused) {
            block_profiler_resume<Cfg>(profiler);
        }
    }
}

__device__ __forceinline__ void arrive_cross_sm_indexed(
    uint32_t* bar, int idx, uint32_t inc = 1) {
    if (bar == nullptr) return;
    arrive_cross_sm(&bar[idx], inc);
}

// ── generic GEMM body ────────────────────────────────────────────────────────
// One CTA executes each scheduled tile. WG0 supplies producer/storer warps and
// WG1/WG2 consume. Policy binds tensor descriptors, dependency barriers, output
// column mapping, and the epilogue store.

struct GemmScheduledTile {
    NmcInstruction inst;
    int tile_m;
    int tile_n;
    int k_begin;
    int k_tiles;
};

template <class Cfg, class GemmCfg>
struct InstructionTileScheduler {
    static constexpr bool WAIT_INPUT_ONCE = true;

    const NmcInstruction& inst;

    __device__ __forceinline__
    InstructionTileScheduler(const NmcInstruction& inst_) : inst(inst_) {}

    __device__ __forceinline__
    void init(const NmcGlobals<Cfg>&, NmcOpSmem<Cfg>&) const {}

    template <NmcRole R>
    __device__ __forceinline__
    bool get_tile(int tile_idx, const NmcGlobals<Cfg>&, NmcOpSmem<Cfg>&, GemmScheduledTile& tile) const {
        constexpr int MAX_ENCODED_TILES = NMC_INSTRUCTION_WIDTH - gemm_field::TILE_IDS;
        const int num_tiles = inst.data[gemm_field::NUM_TILES];
        if (num_tiles <= 0 || num_tiles > MAX_ENCODED_TILES || tile_idx >= num_tiles) return false;

        const int num_splits = GemmCfg::SPLIT_K;
        const int split = inst.data[gemm_field::SPLIT];
        const int k_total = inst.data[gemm_field::K_TILES];
        const int k_per = (k_total + num_splits - 1) / num_splits;
        const int k_begin = split * k_per;
        const int k_end = min(k_begin + k_per, k_total);
        const int k_tiles = k_end - k_begin;
        if (k_tiles <= 0) return false;

        const int coord = inst.data[gemm_field::TILE_IDS + tile_idx];
        tile.inst = inst;
        tile.tile_m = (int)((uint32_t)coord >> 16);
        tile.tile_n = coord & 0xffff;
        tile.k_begin = k_begin;
        tile.k_tiles = k_tiles;
        return true;
    }
};

template <class Cfg, class GemmCfg>
struct MoeQueueTileScheduler {
    static constexpr bool WAIT_INPUT_ONCE = false;

    const NmcInstruction& drain_inst;
    int* task_words;
    int* task_count;
    uint32_t* task_head;

    __device__ __forceinline__
    MoeQueueTileScheduler(
        const NmcInstruction& drain_inst_,
        int* task_words_,
        int* task_count_,
        uint32_t* task_head_)
        : drain_inst(drain_inst_),
          task_words(task_words_),
          task_count(task_count_),
          task_head(task_head_) {}

    static __device__ __forceinline__
    int row_blocks_per_layer(const NmcGlobals<Cfg>& g) {
        static_assert(Cfg::MoeDown::BM >= Cfg::MoeUpGate::BM,
            "Fine-grained MoE barriers require Down BM >= UpGate BM");
        static_assert(Cfg::MoeDown::BM % Cfg::MoeUpGate::BM == 0,
            "Fine-grained MoE barriers require Down BM to be a multiple of UpGate BM");
        constexpr int bm = Cfg::MoeDown::BM;
        const int max_rows = g.BS;
        const int rows_per_expert = ((max_rows + bm - 1) / bm) * bm;
        return NMC_NUM_EXPERTS * (rows_per_expert / bm);
    }

    static __device__ __forceinline__
    int rows_per_expert(const NmcGlobals<Cfg>& g) {
        constexpr int bm = Cfg::MoeDown::BM;
        const int max_rows = g.BS;
        return ((max_rows + bm - 1) / bm) * bm;
    }

    static __device__ __forceinline__
    int down_row_block_from_tile(const NmcGlobals<Cfg>& g, int expert, int tile_m, int tile_bm) {
        const int local_row_begin = tile_m * tile_bm - expert * rows_per_expert(g);
        return local_row_begin / Cfg::MoeDown::BM;
    }

    static __device__ __forceinline__
    int row_block_bar_idx(const NmcGlobals<Cfg>& g, int layer, int expert, int down_row_block) {
        const int blocks_per_layer = row_blocks_per_layer(g);
        const int blocks_per_expert = blocks_per_layer / NMC_NUM_EXPERTS;
        return layer * blocks_per_layer + expert * blocks_per_expert + down_row_block;
    }

    __device__ __forceinline__
    void init(const NmcGlobals<Cfg>&, NmcOpSmem<Cfg>& ops) const {
        if (lane_id() == 0) {
            #pragma unroll
            for (int s = 0; s < NMC_MOE_SCHED_RING; ++s) {
                ops.moe_sched_active[s] = 0;
                ops.moe_sched_epoch[s] = 0;
            }
            ops.moe_sched_batch_base = 0;
        }
        __syncwarp();
    }

    template <NmcRole R>
    __device__ __forceinline__
    bool get_tile(int tile_idx, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops, GemmScheduledTile& tile) const {
        const int drain_layer = drain_inst.data[gemm_field::LAYER];
        const int slot = tile_idx % NMC_MOE_SCHED_RING;
        const int epoch = tile_idx + 1;

        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) {
                ops.moe_sched_batch_base = (int)atomicAdd(&task_head[drain_layer], 1u);
                uint32_t local = (uint32_t)ops.moe_sched_batch_base;
                const bool active = (local < (uint32_t)task_count[drain_layer]);
                if (active) {
                    const MoeTaskRecord rec = load_moe_task(task_words, drain_layer, g.max_moe_tasks, local);
                    NmcInstruction& dst = ops.moe_sched_inst[slot];
                    const bool is_up = (drain_inst.opcode() == NmcOpcode::MOE_UPGATE_ACT_DRAIN);
                    const int k_total = is_up ? (g.D / GemmCfg::BK) : (NMC_EXPERT_DFF / GemmCfg::BK);
                    const int coord = rec.tile_ids;
                    const int tile_m = (int)((uint32_t)coord >> 16);
                    const int expert = rec.layer_expert % NMC_NUM_EXPERTS;  // packed: expert dropped
                    const int tile_bm = is_up ? Cfg::MoeUpGate::BM : Cfg::MoeDown::BM;
                    const int down_row_block = down_row_block_from_tile(g, expert, tile_m, tile_bm);
                    const int row_block_idx = row_block_bar_idx(g, drain_layer, expert, down_row_block);
                    dst.data[gemm_field::OPCODE] = (int)drain_inst.opcode();
                    dst.data[gemm_field::LAYER] = rec.layer_expert;      // expert-layer index
                    dst.data[gemm_field::SPLIT] = rec.split;
                    dst.data[gemm_field::K_TILES] = k_total;
                    dst.data[gemm_field::WAIT_BAR_IDX] = is_up ? drain_layer : row_block_idx;
                    dst.data[gemm_field::WAIT_TARGET] = is_up ? g.BS : rec.wait_target;
                    #ifdef NMC_MOE_FINE_GRAINED_DOWN_COMBINE
                    // Up and down completions use the same (layer, expert,
                    // down-row-block) indexing. This lets MOE_COMBINE wait only
                    // for the eight blocks consumed by its token.
                    dst.data[gemm_field::PRODUCE_BAR_IDX] = row_block_idx;
                    #else
                    // Up still publishes per row-block (the down drain's
                    // WAIT_BAR_IDX above depends on it); down collapses to one
                    // layer-wide counter consumed by wait_moe_down_for_row().
                    dst.data[gemm_field::PRODUCE_BAR_IDX] =
                        is_up ? row_block_idx : drain_layer;
                    #endif
                    dst.data[gemm_field::NUM_TILES] = 1;
                    dst.data[gemm_field::TILE_IDS] = coord;
                }
                *(volatile int*)&ops.moe_sched_active[slot] = active ? 1 : 0;
                __threadfence_block();
                *(volatile int*)&ops.moe_sched_epoch[slot] = epoch;
            }
            __syncwarp();
        } else {
            while (*(volatile int*)&ops.moe_sched_epoch[slot] != epoch) {
                __nanosleep(20);
            }
            __threadfence_block();
        }

        if (*(volatile int*)&ops.moe_sched_active[slot] == 0) return false;

        tile.inst = ops.moe_sched_inst[slot];
        const int coord = tile.inst.data[gemm_field::TILE_IDS];
        const int num_splits = GemmCfg::SPLIT_K;
        const int split = tile.inst.data[gemm_field::SPLIT];
        const int k_total = tile.inst.data[gemm_field::K_TILES];
        const int k_per = (k_total + num_splits - 1) / num_splits;
        const int k_begin = split * k_per;
        const int k_end = min(k_begin + k_per, k_total);
        const int k_tiles = k_end - k_begin;
       
        if (k_tiles <= 0) {
            #ifdef NMC_TRAP_ON_ERROR
            __trap();
            #endif
            return false;
        }

        tile.tile_m = (int)((uint32_t)coord >> 16);
        tile.tile_n = coord & 0xffff;
        tile.k_begin = k_begin;
        tile.k_tiles = k_tiles;
        return true;
    }
};

template <NmcRole R, int CWG_IDX, class Cfg, class GemmCfg, class Policy, class Scheduler>
__device__ void gemm_n8_op_scheduled(
    const NmcInstruction& inst,
    const NmcGlobals<Cfg>& g,
    NmcOpSmem<Cfg>& ops,
    Scheduler scheduler) {
    constexpr int BM      = GemmCfg::BM;
    constexpr int BN      = GemmCfg::BN;
    constexpr int BK      = GemmCfg::BK;
    constexpr int NUM_CWG = GemmCfg::NUM_CWG;
    constexpr int STAGES  = GemmCfg::STAGES;
    constexpr int PREFETCH_STAGES = GemmCfg::PREFETCH_STAGES;
    // CONSUMER warpgroups beyond NUM_CWG are effectively idle (they still
    // hit the same worker_sync calls so the pipeline stays in lockstep).
    constexpr bool consumer_active =
        (R == NmcRole::CONSUMER) && (CWG_IDX < NUM_CWG);
    // Atomic MoE-combine arm forces an smem-staged epilogue even when the
    // tiling asks for DIRECT_STORE: we need y_smem so STORER can issue
    // per-row TMA reduces into x_ffn. Deadlock trap: gemm_store_done_bar
    // arrival count must flip from NUM_CWG (direct) back to 1 (storer).
    constexpr bool direct_store =
        GemmCfg::DIRECT_STORE && !Policy::ATOMIC_COMBINE;

    using w_tile     = st_bf<BN, BK>;
    using x_tile     = st_bf<BM, BK>;
    using y_tile     = st_bf<BM, 64, false>;
    using w_cwg_tile = st_bf<64, BK>;

    extern __shared__ int __shm[];
    // NOTE: GEMM is the heaviest op so its smem layout owns the workspace.
    // Other ops re-use the same dynamic smem region with their own allocator.
    tma_swizzle_allocator al((int*)&__shm[0]);

    w_tile (&w_smem)[STAGES]  = al.allocate<w_tile, STAGES>();
    x_tile (&x_smem)[STAGES]  = al.allocate<x_tile, STAGES>();
    y_tile (&y_smem)[NUM_CWG] = al.allocate<y_tile, NUM_CWG>();

    // Per-op pipeline semaphores live in the kernel-level NmcOpSmem (passed
    // by ref) so all 4 role instantiations share the same smem addresses.
    auto& data_bar  = ops.gemm_data_bar;
    auto& empty_bar = ops.gemm_empty_bar;
    static_assert(STAGES <= MAX_GEMM_STAGES, "STAGES exceeds NmcOpSmem capacity");
    static_assert(PREFETCH_STAGES >= 0 && PREFETCH_STAGES <= STAGES,
        "PREFETCH_STAGES must be in [0, STAGES]");

    // Count empty_bar per consumer warp, not warpgroup. Each warp can retire its
    // own WGMMA group while peers still read the same stage; a per-warpgroup
    // arrival lets the producer reuse that stage early. One ring lap can corrupt
    // data and two can hang on mbarrier parity. Keep this consistent with the
    // big-M path's NUM_CONSUMER_WARPS accounting.
    constexpr int NUM_CONSUMER_WARPS = NUM_CWG * kittens::WARPGROUP_WARPS;

    auto& store_done_bar = ops.gemm_store_done_bar;

    const int num_splits = GemmCfg::SPLIT_K;

    const int lane = lane_id();
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                init_semaphore(data_bar[s],  0,                  1);
                init_semaphore(empty_bar[s], NUM_CONSUMER_WARPS, 0);
            }
            // Direct stores complete per consumer warp; staged stores complete
            // once from the storer. The epilogues must remain mutually exclusive
            // or surplus arrivals will lap store_done_bar's phase.
            constexpr int store_done_arrivals =
                direct_store ? NUM_CONSUMER_WARPS : 1;
            kittens::init_semaphore(store_done_bar, 0, store_done_arrivals);
        }
        __syncwarp();
        scheduler.init(g, ops);
        if constexpr (Scheduler::WAIT_INPUT_ONCE && PREFETCH_STAGES == 0) {
            Policy::wait_input_bars(g, inst, ops); // NOTE: instruction merging assumes all output tiles share the same input barrier.
            if (lane == 0) block_profiler_begin<Cfg>(ops.profiler);
        }
    }
    worker_sync<Cfg>();

    int tile_k_base = 0;
    for (int tile_idx = 0;; ++tile_idx) {
        GemmScheduledTile tile;
        if (!scheduler.template get_tile<R>(tile_idx, g, ops, tile)) break;
        const int layer = tile.inst.data[gemm_field::LAYER];
        const int tile_m = tile.tile_m;
        const int tile_n = tile.tile_n;
        const int k_begin = tile.k_begin;
        const int k_tiles = tile.k_tiles;

        if constexpr (R == NmcRole::PRODUCER) {
            if constexpr (PREFETCH_STAGES > 0) {
                // Open before W prefetch. If the input is not ready, the profiled
                // wait below pauses this range and resumes it before X prefetch.
                if (lane == 0) block_profiler_begin<Cfg>(ops.profiler);
            } else if constexpr (!Scheduler::WAIT_INPUT_ONCE) {
                // Dynamic drains do not open a range until they claim real work.
                // This keeps empty drain instructions off the compact trace.
                if (lane == 0) block_profiler_begin<Cfg>(ops.profiler);
            }
        }

        // ── PRODUCER: TMA loads; data/empty phases continue across tiles ─────
        if constexpr (R == NmcRole::PRODUCER) {
            constexpr bool MOE_L2_HINT = (NMC_MOE_L2_HINT != 0);
            auto load_weight = [&](int stage, int k_abs) {
                if constexpr (MOE_L2_HINT) {
                    warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                        w_smem[stage], Policy::W(g), {0, layer, tile_n, k_abs}, data_bar[stage]);
                } else {
                    warp::tma::load_async(w_smem[stage], Policy::W(g),
                        {0, layer, tile_n, k_abs}, data_bar[stage]);
                }
            };
            auto load_activation = [&](int stage, int k_abs) {
                if constexpr (MOE_L2_HINT) {
                    warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_LAST>(
                        x_smem[stage], Policy::X(g), {0, 0, tile_m, k_abs}, data_bar[stage]);
                } else {
                    warp::tma::load_async(x_smem[stage], Policy::X(g),
                        {0, 0, tile_m, k_abs}, data_bar[stage]);
                }
            };

            if constexpr (PREFETCH_STAGES > 0) {
                // Arm the configured prefix and issue W only. data_bar remains
                // pending because each stage expects both W and X bytes.
                const int num_prefetch = min(PREFETCH_STAGES, k_tiles);
                __syncwarp();
                for (int ki = 0; ki < num_prefetch; ++ki) {
                    const int global_ki = tile_k_base + ki;
                    const int stage = global_ki % STAGES;
                    const int empty_phase = ((global_ki - STAGES) / STAGES) % 2;
                    __syncwarp();
                    if (global_ki >= STAGES) wait(empty_bar[stage], empty_phase);
                    __syncwarp();
                    const int k_abs = k_begin + ki;
                    warp::tma::expect_bytes(
                        data_bar[stage], sizeof(w_tile) + sizeof(x_tile));
                    load_weight(stage, k_abs);
                }

                __syncwarp();
                if constexpr (Scheduler::WAIT_INPUT_ONCE) {
                    // Merged static tiles share one input dependency.
                    if (tile_idx == 0) Policy::wait_input_bars(g, tile.inst, ops);
                } else {
                    // Dynamic MoE records can carry different per-task barriers.
                    Policy::wait_input_bars(g, tile.inst, ops);
                }
                __syncwarp();

                // Complete the armed stages with X. Consumers waiting on data_bar
                // can begin as soon as each corresponding activation TMA arrives.
                for (int ki = 0; ki < num_prefetch; ++ki) {
                    const int global_ki = tile_k_base + ki;
                    const int stage = global_ki % STAGES;
                    const int k_abs = k_begin + ki;
                    load_activation(stage, k_abs);
                }

                for (int ki = num_prefetch; ki < k_tiles; ++ki) {
                    const int global_ki = tile_k_base + ki;
                    const int stage = global_ki % STAGES;
                    const int empty_phase = ((global_ki - STAGES) / STAGES) % 2;
                    __syncwarp();
                    if (global_ki >= STAGES) wait(empty_bar[stage], empty_phase);
                    __syncwarp();
                    const int k_abs = k_begin + ki;
                    warp::tma::expect_bytes(
                        data_bar[stage], sizeof(w_tile) + sizeof(x_tile));
                    load_weight(stage, k_abs);
                    load_activation(stage, k_abs);
                }
            } else {
                // Legacy path. Dynamic MoE drains preserve their original behavior:
                // each producer can issue its first W before the per-task input wait.
                __syncwarp();
                for (int ki = 0; ki < k_tiles; ++ki) {
                    const int global_ki = tile_k_base + ki;
                    const int stage = global_ki % STAGES;
                    const int empty_phase = ((global_ki - STAGES) / STAGES) % 2;
                    __syncwarp();
                    if (global_ki >= STAGES) wait(empty_bar[stage], empty_phase);
                    __syncwarp();
                    const int k_abs = k_begin + ki;
                    warp::tma::expect_bytes(
                        data_bar[stage], sizeof(w_tile) + sizeof(x_tile));
                    load_weight(stage, k_abs);
                    if constexpr (!Scheduler::WAIT_INPUT_ONCE) {
                        if (ki == 0) Policy::wait_input_bars(g, tile.inst, ops);
                    }
                    load_activation(stage, k_abs);
                }
            }
        } else if constexpr (consumer_active) {
            constexpr int cwg_idx = CWG_IDX;

            static_assert(BM == 16, "NMC GEMM requires a 16-row physical tile");
            static_assert(!direct_store || !Policy::FUSE_UPGATE_ACT,
                "NMC fused UpGate must materialize both CWGs before activation");
            gemm_n8::acc_n8x4 acc;
            gemm_n8::zero_n8x4(acc);

            const int tile_k_begin = tile_k_base;
            int stage = tile_k_begin % STAGES;
            int phase = (tile_k_begin / STAGES) % 2;
            int release_stage = stage;

            auto w_cwg_ref = [&](int st) -> w_cwg_tile& {
                bf16* base = reinterpret_cast<bf16*>(&w_smem[st]);
                return *reinterpret_cast<w_cwg_tile*>(base + cwg_idx * 64 * BK);
            };
            __syncwarp();
            wait(data_bar[stage], phase);
            __syncwarp();
            gemm_n8::mma_ABt_n8_independent(acc, w_cwg_ref(stage), x_smem[stage]);
            if (++stage >= STAGES) { stage = 0; phase ^= 1; }

            for (int ki = 1; ki < k_tiles; ++ki) {
                __syncwarp();
                wait(data_bar[stage], phase);
                __syncwarp();
                gemm_n8::mma_ABt_n8_independent(acc, w_cwg_ref(stage), x_smem[stage]);
                warpgroup::mma_async_wait<1>();
                __syncwarp();
                // Each consumer warp releases the stage for itself only, after
                // its own wgmma group has retired. See the empty_bar comment at
                // the top of this function for why this must not be per-CWG.
                if (lane == 0) arrive(empty_bar[release_stage]);
                __syncwarp();
                if (++release_stage >= STAGES) release_stage = 0;
                if (++stage >= STAGES) { stage = 0; phase ^= 1; }
            }
            warpgroup::mma_async_wait();
            __syncwarp();
            if (lane == 0) arrive(empty_bar[release_stage]);
            __syncwarp();

            if (tile_idx > 0) {
                __syncwarp();
                kittens::wait(store_done_bar, (tile_idx - 1) & 1);
                __syncwarp();
            }

            int warp_in_wg_ = warpgroup::warpid();
            bf16* y_ptr     = reinterpret_cast<bf16*>(&y_smem[cwg_idx]);
            if constexpr (direct_store) {
                gemm_n8::store_frag_n8x4_sum_global(
                    Policy::Y(g).raw_ptr, (int)Policy::Y(g).cols(), acc,
                    tile_m, tile_n, cwg_idx, BN, GemmCfg::M_ROWS,
                    num_splits > 1,
                    warp_in_wg_, lane);
                // One arrival per warp; store_done_arrivals must match.
                if (lane == 0) kittens::arrive(store_done_bar);
            } else if constexpr (Policy::ATOMIC_COMBINE) {
                // Stage into y_smem with per-row routed_score scale. STORER
                // later TMA-reduces valid rows into x_ffn. Padding rows get
                // a garbage scale (0) and are skipped by store_y masking.
                const int layer_expert = tile.inst.data[gemm_field::LAYER];
                const int layer = layer_expert / NMC_NUM_EXPERTS;
                const int t0 = 2 * (lane % 4);
                auto score_at = [&](int row_in_tile) -> float {
                    if (row_in_tile >= GemmCfg::M_ROWS) return 0.0f;
                    const int routed = tile_m * BM + row_in_tile;
                    return __bfloat162float(
                        g.routed_scores[nmc_routed_idx(g, layer, routed)]);
                };
                const float s0 = score_at(t0);
                const float s1 = score_at(t0 + 1);
                gemm_n8::store_frag_n8x4_sum_scaled(
                    y_ptr, acc, s0, s1, warp_in_wg_, lane);
            } else {
                gemm_n8::store_frag_n8x4_sum(y_ptr, acc, warp_in_wg_, lane);
            }
            gemm_tma_store_fence();
        }

        worker_sync<Cfg>();

        // The two consumer warpgroups materialize gate and up projections in
        // adjacent shared-memory tiles before cooperatively applying SiLU.
        // Inspired by activation fusion in DeepSeek's MegaMoE
        if constexpr (Policy::FUSE_UPGATE_ACT && NUM_CWG == 2) {
            if constexpr (consumer_active) {
                constexpr int wg_threads   = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
                constexpr int worker_count = 2 * wg_threads;
                const int worker_tid = (int)threadIdx.x - wg_threads;
                constexpr int total = BM * 64;
                constexpr int n_iters = (total + worker_count - 1) / worker_count;
                bf16* gate = reinterpret_cast<bf16*>(&y_smem[0]);
                bf16* up   = reinterpret_cast<bf16*>(&y_smem[1]);
                #pragma unroll
                for (int it = 0; it < n_iters; ++it) {
                    int i = worker_tid + it * worker_count;
                    if (i >= total) break;
                    float gv = __bfloat162float(gate[i]);
                    float uv = __bfloat162float(up[i]);
                    float sv = gv / (1.0f + __expf(-gv));
                    gate[i] = __float2bfloat16(sv * uv);
                }
                gemm_tma_store_fence();
            }
            worker_sync<Cfg>();
        }

        if constexpr (R == NmcRole::STORER) {
            if (lane == 0) {
                NmcInstruction tile_inst = tile.inst;
                tile_inst.data[gemm_field::TILE_M] = tile_m;
                tile_inst.data[gemm_field::TILE_N] = tile_n;
                if (tile.inst.opcode() == NmcOpcode::FFN_UPGATE_ACT) {
                    const int num_n_up_64 = g.Dff / 64;
                    const int silu_per_split = num_n_up_64 / Cfg::Down::SPLIT_K;
                    tile_inst.data[gemm_field::PRODUCE_BAR_IDX] =
                        layer * Cfg::Down::SPLIT_K + (tile_n / silu_per_split);
                }
                if constexpr (!direct_store) {
                    #pragma unroll
                    for (int cwg = 0; cwg < Policy::STORE_TILES; ++cwg) {
                        int row_tile = tile_m;
                        int col_tile = Policy::store_col_tile(tile_n, cwg);
                        Policy::store_y(g, tile_inst, y_smem[cwg], row_tile, col_tile, num_splits);
                        gemm_tma_store_arrive();
                    }
                    gemm_tma_store_wait<0>();
                    kittens::arrive(store_done_bar);
                }
                Policy::signal_output_bar(g, tile_inst);
            }
        }
        tile_k_base += k_tiles;
    }
}

template <NmcRole R, int CWG_IDX, class Cfg, class GemmCfg, class Policy>
__device__ void gemm_n8_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    InstructionTileScheduler<Cfg, GemmCfg> scheduler(inst);
    gemm_n8_op_scheduled<R, CWG_IDX, Cfg, GemmCfg, Policy>(inst, g, ops, scheduler);
}

template <NmcRole R, int CWG_IDX, class Cfg, class GemmCfg, class Policy>
__device__ void gemm_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    gemm_n8_op<R, CWG_IDX, Cfg, GemmCfg, Policy>(inst, g, ops);
}

template <NmcRole R, int CWG_IDX, class Cfg, class GemmCfg, class Policy, class Scheduler>
__device__ void gemm_op_scheduled(
    const NmcInstruction& inst,
    const NmcGlobals<Cfg>& g,
    NmcOpSmem<Cfg>& ops,
    Scheduler scheduler) {
    gemm_n8_op_scheduled<R, CWG_IDX, Cfg, GemmCfg, Policy>(inst, g, ops, scheduler);
}

// Match vLLM's rotary kernel precision boundary. Q/K projection results and
// rotary-cache values are materialized as bf16 before the kernel promotes them
// to fp32 for the multiply/add. The fused decode epilogue otherwise rotates
// raw fp32 WGMMA accumulators with fp32 cache values, bypassing two observable
// bf16 roundings that the prefill Triton kernel already preserves.
__device__ __forceinline__ float nmc_rope_bf16_component(
    float current,
    float peer,
    float cosine,
    float sine,
    bool is_even) {
    current = __bfloat162float(__float2bfloat16(current));
    peer = __bfloat162float(__float2bfloat16(peer));
    cosine = __bfloat162float(__float2bfloat16(cosine));
    sine = __bfloat162float(__float2bfloat16(sine));
    const float current_cos = current * cosine;
    const float peer_sin = peer * sine;
    return is_even ? (current_cos - peer_sin) : (current_cos + peer_sin);
}

// QKV_PROJ — fused GEMM with custom epilogue.
//   gemm_idx ∈ {0=Q, 1=K, 2=V}. Q optionally applies RoPE in-register and
//   writes to q_out (BS, Hq*HD). K applies RoPE and scatters to the paged K
//   cache (via page_table indirection). V scatters to V cache (no RoPE).
//
// Body mirrors gemm_op's producer/consumer pipeline up through the WGMMA
// loop; its custom epilogue scatters directly from registers.
template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void qkv_n8_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    using QKVCfg = typename Cfg::QKV;
    constexpr int BM      = QKVCfg::BM;
    constexpr int BN      = QKVCfg::BN;       // QKV output tile width (set by JIT config)
    constexpr int BK      = QKVCfg::BK;
    constexpr int NUM_CWG = QKVCfg::NUM_CWG;
    constexpr int STAGES  = QKVCfg::STAGES;
    constexpr int PREFETCH_STAGES = QKVCfg::PREFETCH_STAGES;
    constexpr int HALF_HD = HEAD_DIM / 2;     // RoPE: pair across HD/2

    using w_tile     = st_bf<BN, BK>;
    using x_tile     = st_bf<BM, BK>;
    using w_cwg_tile = st_bf<64, BK>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    w_tile (&w_smem)[STAGES] = al.allocate<w_tile, STAGES>();
    x_tile (&x_smem)[STAGES] = al.allocate<x_tile, STAGES>();

    auto& data_bar  = ops.gemm_data_bar;
    auto& empty_bar = ops.gemm_empty_bar;
    static_assert(STAGES <= MAX_GEMM_STAGES, "STAGES exceeds NmcOpSmem capacity");
    static_assert(PREFETCH_STAGES >= 0 && PREFETCH_STAGES <= STAGES,
        "QKV PREFETCH_STAGES must be in [0, STAGES]");

    // Per-warp (not per-CWG) stage release. QKV runs PREFETCH_STAGES == STAGES,
    // so the producer arms the whole ring up front and sits a full lap ahead;
    // a per-CWG release would let one warp free buffers its 3 peers are still
    // reading and wedge them on data_bar parity. See gemm_n8_op_scheduled.
    constexpr int NUM_CONSUMER_WARPS = NUM_CWG * kittens::WARPGROUP_WARPS;

    const int layer    = inst.data[qkv_field::LAYER];
    const int tile_m   = inst.data[qkv_field::TILE_M];
    const int tile_n   = inst.data[qkv_field::TILE_N];   // head/kv_head
    const int gemm_idx = inst.data[qkv_field::GEMM_IDX];
    const int needs_rope = inst.data[qkv_field::NEEDS_ROPE];
    const int k_tiles  = inst.data[qkv_field::K_TILES];
    if (k_tiles <= 0) return;

    const int q_weight_tiles  = (g.Hq  * HEAD_DIM) / BN;
    const int kv_weight_tiles = (g.Hkv * HEAD_DIM) / BN;
    int w_tile_n = tile_n;
    if (gemm_idx == 1 /* K */)      w_tile_n = q_weight_tiles + tile_n;
    else if (gemm_idx == 2 /* V */) w_tile_n = q_weight_tiles + kv_weight_tiles + tile_n;

    constexpr bool consumer_active =
        (R == NmcRole::CONSUMER) && (CWG_IDX < NUM_CWG);

    const int lane = lane_id();
    auto wait_input_bar = [&]() {
        if (g.bar_layer == nullptr) return;
        const int idx = inst.data[qkv_field::WAIT_BAR_IDX];
        const int tgt = inst.data[qkv_field::WAIT_TARGET];
        if (tgt > 0) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_layer, idx, (uint32_t)tgt, ops.profiler);
        }
    };

    if constexpr (R == NmcRole::PRODUCER) {
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                init_semaphore(data_bar[s],  0,                  1);
                init_semaphore(empty_bar[s], NUM_CONSUMER_WARPS, 0);
            }
        }
        __syncwarp();
        if constexpr (PREFETCH_STAGES == 0) {
            wait_input_bar();
            if (lane == 0) block_profiler_begin<Cfg>(ops.profiler);
        }
    }
    worker_sync<Cfg>();

    if (!any_active_row_in_tile(g, ops.row_active_bits, tile_m, BM)) {
        if constexpr (R == NmcRole::STORER) {
            if (lane == 0 && g.bar_qkv != nullptr) {
                int idx = inst.data[qkv_field::PRODUCE_BAR_IDX];
                arrive_cross_sm_indexed(g.bar_qkv, idx, 1u);
            }
        }
        return;
    }

    if constexpr (R == NmcRole::PRODUCER) {
        constexpr int initial_stage_limit =
            PREFETCH_STAGES > 0 ? PREFETCH_STAGES : STAGES;
        const int initial_stages = min(initial_stage_limit, k_tiles);
        if (lane == 0) {
            if constexpr (PREFETCH_STAGES > 0) {
                block_profiler_begin<Cfg>(ops.profiler);
                // Arm each initial stage for W+X, but issue only immutable W before
                // the layer-input dependency. No TMA completion wait is required.
                for (int s = 0; s < initial_stages; ++s) {
                    tma::expect_bytes(data_bar[s], sizeof(w_tile) + sizeof(x_tile));
                    if constexpr (NMC_MOE_L2_HINT != 0) {
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                            w_smem[s], g.W_qkv, {0, layer, w_tile_n, s}, data_bar[s]);
                    } else {
                        tma::load_async(w_smem[s], g.W_qkv,
                            {0, layer, w_tile_n, s}, data_bar[s]);
                    }
                }
            } else {
                for (int s = 0; s < initial_stages; ++s) {
                    tma::expect_bytes(data_bar[s], sizeof(w_tile) + sizeof(x_tile));
                    if constexpr (NMC_MOE_L2_HINT != 0) {
                        // QKV weights stream once -> EVICT_FIRST; x_resid reused -> EVICT_LAST.
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                            w_smem[s], g.W_qkv, {0, layer, w_tile_n, s}, data_bar[s]);
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_LAST>(
                            x_smem[s], g.x_resid, {0, 0, tile_m, s}, data_bar[s]);
                    } else {
                        tma::load_async(w_smem[s], g.W_qkv,
                            {0, layer, w_tile_n, s}, data_bar[s]);
                        tma::load_async(x_smem[s], g.x_resid,
                            {0, 0, tile_m, s}, data_bar[s]);
                    }
                }
            }
        }
        __syncwarp();

        if constexpr (PREFETCH_STAGES > 0) {
            wait_input_bar();
            __syncwarp();
            if (lane == 0) {
                // Finish the armed stages with X. data_bar releases each consumer
                // only after both its earlier W and this activation load arrive.
                for (int s = 0; s < initial_stages; ++s) {
                    if constexpr (NMC_MOE_L2_HINT != 0) {
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_LAST>(
                            x_smem[s], g.x_resid, {0, 0, tile_m, s}, data_bar[s]);
                    } else {
                        tma::load_async(x_smem[s], g.x_resid,
                            {0, 0, tile_m, s}, data_bar[s]);
                    }
                }
            }
            __syncwarp();
        }

        for (int ki = initial_stages; ki < k_tiles; ++ki) {
            int stage = ki % STAGES;
            __syncwarp();
            if (ki >= STAGES) {
                const int phase = ((ki - STAGES) / STAGES) % 2;
                wait(empty_bar[stage], phase);
            }
            __syncwarp();
            warp::tma::expect_bytes(data_bar[stage], sizeof(w_tile) + sizeof(x_tile));
            if constexpr (NMC_MOE_L2_HINT != 0) {
                warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                    w_smem[stage], g.W_qkv, {0, layer, w_tile_n, ki}, data_bar[stage]);
                warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_LAST>(
                    x_smem[stage], g.x_resid, {0, 0, tile_m, ki}, data_bar[stage]);
            } else {
                warp::tma::load_async(w_smem[stage], g.W_qkv,
                    {0, layer, w_tile_n, ki}, data_bar[stage]);
                warp::tma::load_async(x_smem[stage], g.x_resid,
                    {0, 0, tile_m, ki}, data_bar[stage]);
            }
        }
    } else if constexpr (consumer_active) {
        constexpr int cwg_idx = CWG_IDX;

        static_assert(BM == 16, "NMC QKV requires a 16-row physical tile");
        static_assert(QKVCfg::M_ROWS <= 8, "NMC QKV computes up to 8 real rows");
        gemm_n8::acc_n8x4 acc;
        gemm_n8::zero_n8x4(acc);

        int stage = 0, phase = 0;
        int release_stage = 0;

        auto w_cwg_ref = [&](int st) -> w_cwg_tile& {
            bf16* base = reinterpret_cast<bf16*>(&w_smem[st]);
            return *reinterpret_cast<w_cwg_tile*>(base + cwg_idx * 64 * BK);
        };

        __syncwarp();
        wait(data_bar[0], 0);
        __syncwarp();
        gemm_n8::mma_ABt_n8_independent(acc, w_cwg_ref(0), x_smem[0]);
        if (++stage >= STAGES) { stage = 0; phase ^= 1; }

        for (int ki = 1; ki < k_tiles; ++ki) {
            __syncwarp();
            wait(data_bar[stage], phase);
            __syncwarp();
            gemm_n8::mma_ABt_n8_independent(acc, w_cwg_ref(stage), x_smem[stage]);
            warpgroup::mma_async_wait<1>();
            __syncwarp();
            if (lane == 0) arrive(empty_bar[release_stage]);
            __syncwarp();
            if (++release_stage >= STAGES) release_stage = 0;
            if (++stage >= STAGES) { stage = 0; phase ^= 1; }
        }
        warpgroup::mma_async_wait();
        __syncwarp();
        if (lane == 0) arrive(empty_bar[release_stage]);
        __syncwarp();

        // Scatter to gmem.
        const size_t layer_q_stride  = (size_t)g.BS * g.Hq * HEAD_DIM;
        const size_t layer_pt_stride = (size_t)g.BS * g.max_pages_per_seq;
        bf16* q_base = reinterpret_cast<bf16*>(g.q_out.raw_ptr);

        auto scatter_qkv = [&](float val, int hd_pos, int bi) {
            if (bi >= g.BS) return;
            if (!row_is_active(g, ops.row_active_bits, bi)) return;
            bf16 bv = __float2bfloat16(val);
            if (gemm_idx == 0 /* Q */) {
                if (q_base != nullptr) {
                    q_base[(size_t)layer * layer_q_stride
                           + (size_t)bi * g.Hq * HEAD_DIM
                           + tile_n * BN + hd_pos] = bv;
                }
            } else {
                bf16* cache = (gemm_idx == 1) ? g.k_cache : g.v_cache;
                if (cache == nullptr) return;
                int pos     = g.cache_seqlens[bi];
                int page    = pos / g.page_block_size;
                int off     = pos % g.page_block_size;
                int phys    = g.page_table[
                    (size_t)layer * layer_pt_stride
                    + bi * g.max_pages_per_seq + page];
                size_t idx =
                    (size_t)phys * g.page_block_size * g.Hkv * HEAD_DIM
                    + (size_t)off * g.Hkv * HEAD_DIM
                    + (size_t)tile_n * BN + hd_pos;
                cache[idx] = bv;
            }
        };

        // n8 writes four fp32 regs per lane: two token rows at hd_pos n_lo
        // and two token rows at n_hi. Sum the four independent K chunks,
        // then mirror the native QKV RoPE/scatter logic on that compact fragment.
        float qkv_sum[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qkv_sum[i] = acc.part[0].data[i] + acc.part[1].data[i] +
                         acc.part[2].data[i] + acc.part[3].data[i];
        }
        const int warp_row_base = warpgroup::warpid() * 16;
        const int lane_row      = lane / 4;
        const int t0            = 2 * (lane % 4);
        const int n_lo          = cwg_idx * 64 + warp_row_base + lane_row;
        const int n_hi          = n_lo + 8;
        const int b0            = tile_m * BM + t0;
        const int b1            = b0 + 1;
        const bool is_even_row  = (lane_row % 2 == 0);

        if (needs_rope && gemm_idx != 2 /* QIDX_V */ && g.cos_table != nullptr) {
            const int head_col_lo = (tile_n % (HEAD_DIM / BN)) * BN + n_lo;
            const int head_col_hi = (tile_n % (HEAD_DIM / BN)) * BN + n_hi;
            const int freq_lo = head_col_lo / 2;
            const int freq_hi = head_col_hi / 2;
            const float peer0 = __shfl_xor_sync(0xffffffff, qkv_sum[0], 4);
            const float peer1 = __shfl_xor_sync(0xffffffff, qkv_sum[1], 4);
            const float peer2 = __shfl_xor_sync(0xffffffff, qkv_sum[2], 4);
            const float peer3 = __shfl_xor_sync(0xffffffff, qkv_sum[3], 4);
            if (b0 < g.BS && row_is_active(g, ops.row_active_bits, b0)) {
                int pos = g.cache_seqlens[b0];
                float c = g.cos_table[(size_t)pos * HALF_HD + freq_lo];
                float s = g.sin_table[(size_t)pos * HALF_HD + freq_lo];
                qkv_sum[0] = nmc_rope_bf16_component(
                    qkv_sum[0], peer0, c, s, is_even_row);
                c = g.cos_table[(size_t)pos * HALF_HD + freq_hi];
                s = g.sin_table[(size_t)pos * HALF_HD + freq_hi];
                qkv_sum[2] = nmc_rope_bf16_component(
                    qkv_sum[2], peer2, c, s, is_even_row);
            }
            if (b1 < g.BS && row_is_active(g, ops.row_active_bits, b1)) {
                int pos = g.cache_seqlens[b1];
                float c = g.cos_table[(size_t)pos * HALF_HD + freq_lo];
                float s = g.sin_table[(size_t)pos * HALF_HD + freq_lo];
                qkv_sum[1] = nmc_rope_bf16_component(
                    qkv_sum[1], peer1, c, s, is_even_row);
                c = g.cos_table[(size_t)pos * HALF_HD + freq_hi];
                s = g.sin_table[(size_t)pos * HALF_HD + freq_hi];
                qkv_sum[3] = nmc_rope_bf16_component(
                    qkv_sum[3], peer3, c, s, is_even_row);
            }
        }
        scatter_qkv(qkv_sum[0], n_lo, b0);
        scatter_qkv(qkv_sum[1], n_lo, b1);
        scatter_qkv(qkv_sum[2], n_hi, b0);
        scatter_qkv(qkv_sum[3], n_hi, b1);
        __threadfence();
    }

    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane == 0 && g.bar_qkv != nullptr) {
            int idx = inst.data[qkv_field::PRODUCE_BAR_IDX];
            arrive_cross_sm_indexed(g.bar_qkv, idx, 1u);
        }
    }
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void qkv_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    qkv_n8_op<R, CWG_IDX, Cfg>(inst, g, ops);
}

// Column masking helpers for the first and last partially visible KV blocks.
__device__ __forceinline__ void
mask_att_cols(rt_fl<16, ATTN_BLOCK_KV>& att, int valid_cols) {
    int lane = kittens::laneid();
    #pragma unroll
    for (int j = 0; j < ATTN_BLOCK_KV / 16; j++) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int col = j * 16 + (k / 2) * 8 + (lane % 4) * 2;
            if (col     >= valid_cols) att.tiles[0][j].data[k].x = -INFINITY;
            if (col + 1 >= valid_cols) att.tiles[0][j].data[k].y = -INFINITY;
        }
    }
}
__device__ __forceinline__ void
mask_att_cols_lo(rt_fl<16, ATTN_BLOCK_KV>& att, int min_col) {
    int lane = kittens::laneid();
    #pragma unroll
    for (int j = 0; j < ATTN_BLOCK_KV / 16; j++) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int col = j * 16 + (k / 2) * 8 + (lane % 4) * 2;
            if (col     < min_col) att.tiles[0][j].data[k].x = -INFINITY;
            if (col + 1 < min_col) att.tiles[0][j].data[k].y = -INFINITY;
        }
    }
}

// ATTN_DECODE — paged GQA decode with split-KV support. WG0 warp 1 produces
// staged K/V tiles, WG1 consumes them, and WG2 remains idle. Output goes to
// o_partial/lse_partial for multiple splits or directly to o_proj_in for one.
template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void attn_decode(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    // ATTN_DECODE uses WG1 only; WG2 stays idle even when NUM_CWG==2.
    constexpr bool consumer_active = (R == NmcRole::CONSUMER) && (CWG_IDX == 0);
    // Release K/V stages per consumer warp. A per-warpgroup arrival can free
    // shared stages while peer warps still read them; see the GEMM empty_bar
    // invariant in gemm_n8_op_scheduled.
    constexpr int ATTN_CONSUMER_WARPS = kittens::WARPGROUP_WARPS;
    using k_tile = typename NmcGlobals<Cfg>::k_tile;
    using v_tile = typename NmcGlobals<Cfg>::v_tile;
    using q_smem_tile = st_bf<Q_TILE_H, HEAD_DIM, false>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    q_smem_tile& q_smem               = al.allocate<q_smem_tile>();
    k_tile     (&k_smem)[ATTN_NUM_STAGES] = al.allocate<k_tile, ATTN_NUM_STAGES>();
    v_tile     (&v_smem)[ATTN_NUM_STAGES] = al.allocate<v_tile, ATTN_NUM_STAGES>();

    const int layer       = inst.data[attn_field::LAYER];
    const int split_idx   = inst.data[attn_field::SPLIT_IDX];
    const int num_splits  = inst.data[attn_field::NUM_SPLITS];
    const int window_size = inst.data[attn_field::WINDOW_SIZE];
    const int hr          = g.Hq / g.Hkv;
    const int num_tiles   = inst.data[attn_field::NUM_TILES];
    constexpr int MAX_ENCODED_TILES = NMC_INSTRUCTION_WIDTH - attn_field::TILE_IDS;
    if (num_splits <= 0 || num_tiles <= 0 || num_tiles > MAX_ENCODED_TILES) return;

    auto& k_bar     = ops.attn_k_bar;
    auto& v_bar     = ops.attn_v_bar;
    auto& empty_bar = ops.attn_empty_bar;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int packed_coord = inst.data[attn_field::TILE_IDS + tile_idx];
        const int batch_idx   = (int)((uint32_t)packed_coord >> 16);
        const int kv_head_idx = packed_coord & 0xffff;

        const bool tile_active = row_is_active(g, ops.row_active_bits, batch_idx);
        // cache_seqlens stores the decode token's position before the QKV op
        // appends K/V at that slot. Attention must include the newly written
        // token, matching the PyTorch reference's cache_seqlens + 1 call.
        int seq_len       = tile_active ? (g.cache_seqlens[batch_idx] + 1) : 0;
        int num_kv_blocks = (seq_len + ATTN_BLOCK_KV - 1) / ATTN_BLOCK_KV;

        int window_start = 0, kv_lo_block = 0;
        if (window_size > 0) {
            // Match FlashAttention's left-window semantics for window_size=(W, 0):
            // a decode query at absolute position pos attends [pos - W, pos],
            // i.e. W + 1 tokens after the just-written decode KV is included.
            const int keep_tokens = window_size + 1;
            if (seq_len > keep_tokens) window_start = seq_len - keep_tokens;
            kv_lo_block  = window_start / ATTN_BLOCK_KV;
        }
        int total_window_blocks = num_kv_blocks - kv_lo_block;
        int blocks_per_split    = (total_window_blocks + num_splits - 1) / num_splits;
        int kv_start    = kv_lo_block + split_idx * blocks_per_split;
        int kv_end      = min(kv_start + blocks_per_split, num_kv_blocks);
        int local_count = max(0, kv_end - kv_start);


        const size_t bh_offset      = ((size_t)batch_idx * g.Hkv + kv_head_idx) * hr;
        const size_t o_split_stride = (size_t)g.BS * g.Hkv * hr * HEAD_DIM;
        const size_t l_split_stride = (size_t)g.BS * g.Hkv * hr;
        const bool one_split = (num_splits == 1);

        bf16*  o_dst;
        float* l_dst = (g.lse_partial != nullptr)
                    ? (g.lse_partial + (size_t)split_idx * l_split_stride + bh_offset)
                    : nullptr;
        if (one_split) {
            bf16* o_base = reinterpret_cast<bf16*>(g.o_proj_in.raw_ptr);
            o_dst = o_base + bh_offset * HEAD_DIM;
        } else {
            o_dst = (g.o_partial != nullptr)
                ? (g.o_partial + (size_t)split_idx * o_split_stride + bh_offset * HEAD_DIM)
                : nullptr;
        }

        if (!tile_active || local_count == 0) {
            // The controller warp does not enter dispatch. Index from the first
            // worker thread so small arrays (notably hr <= 16 LSE values) are
            // not skipped by starting at absolute threadIdx.x == 32.
            constexpr int worker_threads =
                Cfg::NUM_THREADS - kittens::WARP_THREADS;
            const int worker_tid =
                static_cast<int>(threadIdx.x) - kittens::WARP_THREADS;
            if (!one_split && l_dst != nullptr) {
                for (int i = worker_tid; i < hr; i += worker_threads)
                    l_dst[i] = -INFINITY;
            }
            if (o_dst != nullptr) {
                for (int i = worker_tid;
                     i < hr * HEAD_DIM;
                     i += worker_threads)
                    o_dst[i] = bf16{};
            }
            worker_sync<Cfg>();
            if constexpr (R == NmcRole::STORER) {
                if (lane_id() == 0) {
                    if (num_splits > 1) {
                        if (g.bar_combine != nullptr) {
                            const int idx = layer * (g.BS * g.Hkv) + batch_idx * g.Hkv + kv_head_idx;
                            arrive_cross_sm_indexed(g.bar_combine, idx, 1u);
                        }
                    } else {
                        if (g.bar_attn != nullptr) {
                            const int kvh_per_op = max(1, g.Hkv / Cfg::OProj::SPLIT_K);
                            const int idx = layer * Cfg::OProj::SPLIT_K + (kv_head_idx / kvh_per_op);
                            arrive_cross_sm_indexed(g.bar_attn, idx, 1u);
                        }
                    }
                }
            }
            worker_sync<Cfg>();
            continue;
        }

        const size_t q_layer_stride = (size_t)g.BS * g.Hq * HEAD_DIM;
        bf16* q_base = reinterpret_cast<bf16*>(g.q_out.raw_ptr);
        bf16* q_src  = q_base + (size_t)layer * q_layer_stride
                    + ((size_t)batch_idx * g.Hkv + kv_head_idx) * hr * HEAD_DIM;
        int q_valid = hr * HEAD_DIM;

        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) {
                #pragma unroll
                for (int s = 0; s < ATTN_NUM_STAGES; s++) {
                    init_semaphore(k_bar[s],     0, 1);
                    init_semaphore(v_bar[s],     0, 1);
                    init_semaphore(empty_bar[s], ATTN_CONSUMER_WARPS, 0);
                }
                if (g.bar_qkv != nullptr) {
                    const int idx = layer * g.Hkv + kv_head_idx;
                    const int tgt = inst.data[attn_field::WAIT_TARGET];
                    if (tgt > 0) {
                        profiled_wait_cross_sm_indexed<Cfg>(
                            g.bar_qkv, idx, (uint32_t)tgt, ops.profiler);
                    }
                }
                block_profiler_begin<Cfg>(ops.profiler);
            }
        }
        worker_sync<Cfg>();

        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) {
                int prefetch = min(ATTN_NUM_STAGES, local_count);
                for (int s = 0; s < prefetch; s++) {
                    int kv_idx    = kv_start + s;
                    int phys_page = g.page_table[
                        (size_t)layer * g.BS * g.max_pages_per_seq
                        + batch_idx * g.max_pages_per_seq + kv_idx];
                    #if NMC_TRAP_ON_ERROR
                    if (phys_page == -1) {
                        __trap();
                    }
                    #endif
                    coord<k_tile> kc = {0, phys_page, 0, kv_head_idx};
                    tma::expect_bytes(k_bar[s], sizeof(k_tile));
                    tma::expect_bytes(v_bar[s], sizeof(v_tile));
                    if constexpr (NMC_MOE_L2_HINT != 0) {
                        // KV cache is read once per decode step and is far larger
                        // than L2 at long context -> stream it (EVICT_FIRST) so it
                        // does not evict the small reused weights/activations of
                        // concurrent ops.
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                            k_smem[s], g.K, kc, k_bar[s]);
                        tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                            v_smem[s], g.V, kc, v_bar[s]);
                    } else {
                        tma::load_async(k_smem[s], g.K, kc, k_bar[s]);
                        tma::load_async(v_smem[s], g.V, kc, v_bar[s]);
                    }
                }
            }
        } else if constexpr (consumer_active) {
            // Cooperative Q load overlaps the producer's initial K/V TMA prefetch.
            for (int i = (int)threadIdx.x % (kittens::WARPGROUP_WARPS * kittens::WARP_THREADS);
                i < Q_TILE_H * HEAD_DIM;
                i += kittens::WARPGROUP_WARPS * kittens::WARP_THREADS) {
                q_smem.data[i] = (i < q_valid) ? q_src[i] : bf16{};
            }
            warpgroup::sync(4);
        }
        worker_sync<Cfg>();

        if constexpr (R == NmcRole::PRODUCER) {
            for (int local_idx = ATTN_NUM_STAGES; local_idx < local_count; local_idx++) {
                int stage = local_idx % ATTN_NUM_STAGES;
                wait(empty_bar[stage], ((local_idx - ATTN_NUM_STAGES) / ATTN_NUM_STAGES) % 2);

                int kv_idx    = kv_start + local_idx;
                int phys_page = g.page_table[
                    (size_t)layer * g.BS * g.max_pages_per_seq
                    + batch_idx * g.max_pages_per_seq + kv_idx];
                #if NMC_TRAP_ON_ERROR
                if (phys_page == -1) {
                    __trap();
                }
                #endif
                coord<k_tile> kc = {0, phys_page, 0, kv_head_idx};
                warp::tma::expect_bytes(k_bar[stage], sizeof(k_tile));
                warp::tma::expect_bytes(v_bar[stage], sizeof(v_tile));
                if constexpr (NMC_MOE_L2_HINT != 0) {
                    warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                        k_smem[stage], g.K, kc, k_bar[stage]);
                    warp::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                        v_smem[stage], g.V, kc, v_bar[stage]);
                } else {
                    warp::tma::load_async(k_smem[stage], g.K, kc, k_bar[stage]);
                    warp::tma::load_async(v_smem[stage], g.V, kc, v_bar[stage]);
                }
            }
        } else if constexpr (consumer_active) {
            rt_bf<16, HEAD_DIM>       q_reg;
            rt_fl<16, ATTN_BLOCK_KV>  att_block;
            rt_bf<16, ATTN_BLOCK_KV>  att_block_mma;
            rt_fl<16, HEAD_DIM>       o_reg;

            // Keep the running maximum in the same log2-scaled domain as the
            // exponent input. This avoids scaling the old and new maxima on
            // every KV tile.
            col_vec<rt_fl<16, ATTN_BLOCK_KV>> max_vec, norm_vec, max_vec_last;

            constexpr float LOG2E_SCALE = 1.44269504089f * 0.08838834764f;
            warp::neg_infty(max_vec);
            warp::zero(norm_vec);
            warp::zero(o_reg);
            warp::load(q_reg, q_smem);

            for (int local_idx = 0; local_idx < local_count; local_idx++) {
                int kv_idx = kv_start + local_idx;
                int stage  = local_idx % ATTN_NUM_STAGES;
                int phase  = (local_idx / ATTN_NUM_STAGES) % 2;

                wait(k_bar[stage], phase);
                warpgroup::mm_ABt(att_block, q_reg, k_smem[stage]);

                warpgroup::mma_async_wait();

                if (kv_idx == num_kv_blocks - 1) {
                    int valid_cols = seq_len - kv_idx * ATTN_BLOCK_KV;
                    if (valid_cols < ATTN_BLOCK_KV) mask_att_cols(att_block, valid_cols);
                }
                if (window_size > 0 && kv_idx == kv_lo_block && window_start > 0) {
                    int min_col = window_start - kv_lo_block * ATTN_BLOCK_KV;
                    if (min_col > 0) mask_att_cols_lo(att_block, min_col);
                }

                warp::mul(att_block, att_block, LOG2E_SCALE);
                warp::copy(max_vec_last, max_vec);
                warp::row_max(max_vec, att_block, max_vec);
                warp::sub_row(att_block, att_block, max_vec);
                warp::exp2(att_block, att_block);
                warp::sub(max_vec_last, max_vec_last, max_vec);
                warp::exp2(max_vec_last, max_vec_last);
                warp::mul(norm_vec, norm_vec, max_vec_last);
                warp::row_sum(norm_vec, att_block, norm_vec);
                warp::copy(att_block_mma, att_block);
                warp::mul_row(o_reg, o_reg, max_vec_last);

                wait(v_bar[stage], phase);
                warpgroup::mma_AB(o_reg, att_block_mma, v_smem[stage]);
                warpgroup::mma_async_wait();
                // One arrival per warp; ATTN_CONSUMER_WARPS must match.
                if (lane_id() == 0) arrive(empty_bar[stage]);
            }

            // LSE write (only when split; first warp of consumer WG owns it).
            if (!one_split && l_dst != nullptr && warpgroup::warpid() == 0) {
                // max_vec is qk * inv_sqrt(D) * log2(e); convert the maximum
                // back to natural-log units for the cross-split LSE.
                constexpr float LN2 = 0.69314718056f;
                int lane = kittens::laneid();
                if (lane % 4 == 0) {
                    int row_top = lane / 4;
                    int row_bot = row_top + 8;
                    float m_top = max_vec.data[0][0].x;
                    float m_bot = max_vec.data[0][0].y;
                    float n_top = norm_vec.data[0][0].x;
                    float n_bot = norm_vec.data[0][0].y;
                    float lse_top = (n_top > 0.f) ? (m_top * LN2 + __logf(n_top)) : -INFINITY;
                    float lse_bot = (n_bot > 0.f) ? (m_bot * LN2 + __logf(n_bot)) : -INFINITY;
                    if (row_top < hr) l_dst[row_top] = lse_top;
                    if (row_bot < hr) l_dst[row_bot] = lse_bot;
                }
            }

            warp::div_row(o_reg, o_reg, norm_vec);

            rt_bf<16, HEAD_DIM> o_bf;
            warp::copy(o_bf, o_reg);

            auto& o_smem = reinterpret_cast<q_smem_tile&>(q_smem);
            if (warpgroup::warpid() == 0) warp::store(o_smem, o_bf);
            warpgroup::sync(5);

            if (o_dst != nullptr) {
                constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
                int local_tid = (int)threadIdx.x % wg_threads;
                for (int i = local_tid; i < q_valid; i += wg_threads)
                    o_dst[i] = o_smem.data[i];
            }
        }

        worker_sync<Cfg>();
        if constexpr (R == NmcRole::STORER) {
            if (lane_id() == 0) {
                if (num_splits > 1) {
                    if (g.bar_combine != nullptr) {
                        const int idx = layer * (g.BS * g.Hkv) + batch_idx * g.Hkv + kv_head_idx;
                        arrive_cross_sm_indexed(g.bar_combine, idx, 1u);
                    }
                } else {
                    if (g.bar_attn != nullptr) {
                        const int kvh_per_op = max(1, g.Hkv / Cfg::OProj::SPLIT_K);
                        const int idx = layer * Cfg::OProj::SPLIT_K + (kv_head_idx / kvh_per_op);
                        arrive_cross_sm_indexed(g.bar_attn, idx, 1u);
                    }
                }
            }
        }
        worker_sync<Cfg>();
    }
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void attn_drain(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[attn_drain_field::LAYER];
    const int queue_offset = inst.data[attn_drain_field::QUEUE_OFFSET];
    // QUEUE_LEN == ATTN_DYNAMIC_SENTINEL => use the per-step host-written
    // g.attn_queue_len. A baked positive length is still accepted for older
    // schedules that stamp the length into the instruction word.
    const int baked_len = inst.data[attn_drain_field::QUEUE_LEN];
    const int queue_len = (baked_len == ATTN_DYNAMIC_SENTINEL)
        ? g.attn_queue_len : baked_len;
    if (g.attn_queue_words == nullptr || g.attn_queue_heads == nullptr ||
        queue_len <= 0 || layer < 0 || layer >= g.num_layers) {
        worker_sync<Cfg>();
        return;
    }

    while (true) {
        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) {
                const uint32_t local = atomicAdd(&g.attn_queue_heads[layer], 1u);
                ops.attn_claim_active = (local < (uint32_t)queue_len) ? 1 : 0;
                if (ops.attn_claim_active) {
                    const int* src = g.attn_queue_words
                        + ((int64_t)queue_offset + (int64_t)local) * NMC_INSTRUCTION_WIDTH;
                    #pragma unroll
                    for (int i = 0; i < NMC_INSTRUCTION_WIDTH; ++i) {
                        ops.attn_claim_inst.data[i] = src[i];
                    }
                    // Full layers share layer-0 queue words; patching LAYER
                    // retargets every barrier, page, and query index.
                    ops.attn_claim_inst.data[attn_field::LAYER] = layer;
                }
            }
        }
        worker_sync<Cfg>();
        if (ops.attn_claim_active == 0) break;
        attn_decode<R, CWG_IDX, Cfg>(ops.attn_claim_inst, g, ops);
        worker_sync<Cfg>();
    }
}

// ATTN_COMBINE — one CTA per (batch, kv_head) combines split-KV partials into
// row-major o_proj_in (BS, Hq*HD).
template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void attn_combine(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int batch_idx   = inst.data[combine_field::BATCH_IDX];
    const int kv_head_idx = inst.data[combine_field::KV_HEAD_IDX];

    if (batch_idx < 0 || batch_idx >= g.BS ||
        kv_head_idx < 0 || kv_head_idx >= g.Hkv ||
        g.Hkv <= 0 || g.Hq <= 0 || g.Hq % g.Hkv != 0 ||
        g.Hq / g.Hkv > 2 * kittens::WARPGROUP_WARPS) {
        #if NMC_TRAP_ON_ERROR
        __trap();
        #endif
        return;
    }
    // Dynamic schedules read reduction/wait bounds from attn_num_splits;
    // static schedules bake a positive count.
    const int baked_splits = inst.data[combine_field::NUM_SPLITS];
    int num_splits = baked_splits;
    int wait_target = inst.data[combine_field::WAIT_TARGET];
    if (baked_splits == ATTN_DYNAMIC_SENTINEL) {
        // Host policy floors S_b >= 2, so an unconditional combine wave is
        // always paired with a real wait. Reading a missing pointer or a
        // non-positive S would hang/corrupt; trap rather than silently proceed.
        if (g.attn_num_splits == nullptr) {
            #if NMC_TRAP_ON_ERROR
            __trap();
            #endif
            return;
        }
        num_splits = g.attn_num_splits[batch_idx];
        wait_target = num_splits;
    }
    if (num_splits <= 0 || num_splits > g.num_splits) {
        #if NMC_TRAP_ON_ERROR
        __trap();
        #endif
        return;
    }
    const int hr          = g.Hq / g.Hkv;
    const bool active     = row_is_active(g, ops.row_active_bits, batch_idx);

    if (g.o_partial == nullptr || g.lse_partial == nullptr) {
        #if NMC_TRAP_ON_ERROR
        __trap();
        #endif
        return;
    }

    const size_t bh = ((size_t)batch_idx * g.Hkv + kv_head_idx) * hr;
    const size_t o_split_stride = (size_t)g.BS * g.Hkv * hr * HEAD_DIM;
    const size_t l_split_stride = (size_t)g.BS * g.Hkv * hr;

    bf16* o_out = reinterpret_cast<bf16*>(g.o_proj_in.raw_ptr);
    if (o_out == nullptr) {
        #if NMC_TRAP_ON_ERROR
        __trap();
        #endif
        return;
    }

    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0 && g.bar_combine != nullptr) {
            int idx = inst.data[combine_field::WAIT_BAR_IDX];
            if (wait_target > 0) {
                profiled_wait_cross_sm_indexed<Cfg>(
                    g.bar_combine, idx, (uint32_t)wait_target, ops.profiler);
            }
        }
        if (lane_id() == 0) block_profiler_begin<Cfg>(ops.profiler);
    }
    worker_sync<Cfg>();

    // One consumer warp per query head computes split weights once and
    // broadcasts them while lanes accumulate output dimensions.
    if constexpr (R == NmcRole::CONSUMER) {
        constexpr int WARPS_PER_WG = kittens::WARPGROUP_WARPS;
        int r = CWG_IDX * WARPS_PER_WG + warpgroup::warpid();
        int lane = kittens::laneid();
        if (r < hr) {
            size_t head_offset = bh + r;
            if (!active) {
                #pragma unroll
                for (int i = 0; i < HEAD_DIM / kittens::WARP_THREADS; i++) {
                    int d = lane + i * kittens::WARP_THREADS;
                    o_out[head_offset * HEAD_DIM + d] = bf16{};
                }
            } else {
                float lse_max = -INFINITY;
                if (lane == 0) {
                    for (int s = 0; s < num_splits; s++) {
                        float lse = g.lse_partial[
                            s * l_split_stride + head_offset];
                        lse_max = fmaxf(lse_max, lse);
                    }
                }
                lse_max = __shfl_sync(0xffffffff, lse_max, 0);

                float acc[HEAD_DIM / kittens::WARP_THREADS] = {};
                float denom = 0.f;
                for (int s = 0; s < num_splits; s++) {
                    float w = 0.f;
                    if (lane == 0) {
                        float lse = g.lse_partial[
                            s * l_split_stride + head_offset];
                        w = __expf(lse - lse_max);
                        denom += w;
                    }
                    w = __shfl_sync(0xffffffff, w, 0);
                    #pragma unroll
                    for (int i = 0; i < HEAD_DIM / kittens::WARP_THREADS; i++) {
                        int d = lane + i * kittens::WARP_THREADS;
                        float oval = __bfloat162float(
                            g.o_partial[s * o_split_stride
                                + head_offset * HEAD_DIM + d]);
                        acc[i] += w * oval;
                    }
                }
                denom = __shfl_sync(0xffffffff, denom, 0);

                #pragma unroll
                for (int i = 0; i < HEAD_DIM / kittens::WARP_THREADS; i++) {
                    int d = lane + i * kittens::WARP_THREADS;
                    o_out[head_offset * HEAD_DIM + d] =
                        __float2bfloat16(acc[i] / denom);
                }
            }
        }
    }

    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0 && g.bar_attn != nullptr) {
            int idx = inst.data[combine_field::PRODUCE_BAR_IDX];
            arrive_cross_sm_indexed(g.bar_attn, idx, 1u);
        }
    }
}

// ── GEMM policies ────────────────────────────────────────────────────────────
// Each policy binds the input/output gls and the cross-SM bar plumbing for a
// specific op. QKV uses its dedicated path above because its epilogue applies
// RoPE and appends K/V through paged-cache indirection.
//
// All policies skip cross-SM bar work when the corresponding pointer is null,
// so single-op tests can drive `gemm_op` without setting up the full barrier
// graph.

// Generic instruction-driven cross-SM bar plumbing — used by every GEMM policy.
// Producer waits on `in_bar[WAIT_BAR_IDX] >= WAIT_TARGET`, then arrives on
// `out_bar[PRODUCE_BAR_IDX] += 1`. Either bar may be nullptr → skip.
template <class Cfg>
__device__ __forceinline__
void inst_wait_in(const uint32_t* in_bar, const NmcInstruction& inst,
                  NmcOpSmem<Cfg>& ops) {
    const int idx    = inst.data[gemm_field::WAIT_BAR_IDX];
    const int target = inst.data[gemm_field::WAIT_TARGET];
    if (target <= 0) return;
    profiled_wait_cross_sm_indexed<Cfg>(
        in_bar, idx, (uint32_t)target, ops.profiler);
}
__device__ __forceinline__
void inst_arrive_out(uint32_t* out_bar, const NmcInstruction& inst) {
    const int idx = inst.data[gemm_field::PRODUCE_BAR_IDX];
    // asm volatile("fence.proxy.async;\n" ::: "memory");
    // __threadfence();
    arrive_cross_sm_indexed(out_bar, idx, 1u);
}

template <class Cfg>
struct UpGateActPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_upgate; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.x_resid; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.silu_out; }
    static constexpr bool FUSE_UPGATE_ACT = true;
    static constexpr int  STORE_TILES = 1;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_layer, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int /*cwg*/) {
        return tile_n;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction&,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int /*num_splits*/) {
        tma::store_async(g.silu_out, y, {0, 0, row_tile, col_tile});
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_silu, inst);
    }
};

template <class Cfg>
struct DownPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_down; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.silu_out; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.x_ffn_gl; }
    static constexpr bool FUSE_UPGATE_ACT = false;
    static constexpr int  STORE_TILES = Cfg::Down::NUM_CWG;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_silu, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int cwg) {
        return tile_n * (Cfg::Down::BN / 64) + cwg;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction& inst,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int num_splits) {
        (void)inst;
        if (num_splits > 1) {
            tma::store_add_async(g.x_ffn_gl, y, {0, 0, row_tile, col_tile});
        } else {
            tma::store_async(g.x_ffn_gl, y, {0, 0, row_tile, col_tile});
        }
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_ffn_down, inst);
    }
};

template <class Cfg>
struct OProjPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_oproj; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.o_proj_in; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.x_attn_gl; }
    static constexpr bool FUSE_UPGATE_ACT = false;
    static constexpr int  STORE_TILES = Cfg::OProj::NUM_CWG;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_attn, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int cwg) {
        return tile_n * (Cfg::OProj::BN / 64) + cwg;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction& inst,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int num_splits) {
        (void)inst;
        if (num_splits > 1) {
            tma::store_add_async(g.x_attn_gl, y, {0, 0, row_tile, col_tile});
        } else {
            tma::store_async(g.x_attn_gl, y, {0, 0, row_tile, col_tile});
        }
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_oproj, inst);
    }
};

template <class Cfg>
struct LMHeadPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_lmhead; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.x_resid; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.lm_logits; }
    static constexpr bool FUSE_UPGATE_ACT = false;
    static constexpr int  STORE_TILES = Cfg::LMHead::NUM_CWG;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        // Final layer's residual; encoder points WAIT_BAR_IDX at the right
        // bar_layer slot. nullptr-skip handles single-op tests.
        inst_wait_in<Cfg>(g.bar_layer, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int cwg) {
        return tile_n * (Cfg::LMHead::BN / 64) + cwg;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction&,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int num_splits) {
        if (num_splits > 1) {
            tma::store_add_async(g.lm_logits, y, {0, 0, row_tile, col_tile});
        } else {
            tma::store_async(g.lm_logits, y, {0, 0, row_tile, col_tile});
        }
    }
    static __device__ __forceinline__
    void signal_output_bar(const G&, const NmcInstruction&) {
        // LM_HEAD is the last op; nothing downstream waits on it.
    }
};

template <class Cfg>
struct RouterPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_router; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.x_resid; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.router_logits; }
    static constexpr bool FUSE_UPGATE_ACT = false;
    static constexpr int STORE_TILES = Cfg::Router::NUM_CWG;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_layer, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int cwg) {
        return tile_n * (Cfg::Router::BN / 64) + cwg;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction& inst,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int num_splits) {
        (void)inst;
        if (num_splits > 1) {
            tma::store_add_async(g.router_logits, y, {0, 0, row_tile, col_tile});
        } else {
            tma::store_async(g.router_logits, y, {0, 0, row_tile, col_tile});
        }
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_router, inst);
    }
};

template <class Cfg>
struct MoeUpGateActPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_moe_upgate; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.moe_x; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.moe_hidden; }
    static constexpr bool FUSE_UPGATE_ACT = true;
    static constexpr int STORE_TILES = 1;
    static constexpr bool ATOMIC_COMBINE = false;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_gather, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int /*cwg*/) {
        return tile_n;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction&,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int /*num_splits*/) {
        tma::store_async(g.moe_hidden, y, {0, 0, row_tile, col_tile});
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_moe_upgate, inst);
    }
};

template <class Cfg>
struct MoeDownPolicy {
    using G = NmcGlobals<Cfg>;
    static __device__ __forceinline__ const auto& W(const G& g) { return g.W_moe_down; }
    static __device__ __forceinline__ const auto& X(const G& g) { return g.moe_hidden; }
    static __device__ __forceinline__ const auto& Y(const G& g) { return g.moe_down_out; }
    static constexpr bool FUSE_UPGATE_ACT = false;
    static constexpr int STORE_TILES = Cfg::MoeDown::NUM_CWG;
    // A/B arm: skip MOE_COMBINE; scale + TMA-reduce into x_ffn instead.
    static constexpr bool ATOMIC_COMBINE = Cfg::MOE_COMBINE_ATOMIC_TMA;

    static __device__ __forceinline__
    void wait_input_bars(const G& g, const NmcInstruction& inst, NmcOpSmem<Cfg>& ops) {
        inst_wait_in<Cfg>(g.bar_moe_upgate, inst, ops);
    }
    static __device__ __forceinline__
    int store_col_tile(int tile_n, int cwg) {
        return tile_n * (Cfg::MoeDown::BN / 64) + cwg;
    }
    static __device__ __forceinline__
    void store_y(const G& g, const NmcInstruction& inst,
                 typename G::act_y_tile& y, int row_tile, int col_tile, int num_splits) {
        if constexpr (ATOMIC_COMBINE) {
            // Scatter-reduce each VALID tile row into x_ffn[token]. Padding
            // rows (slot >= expert_counts) MUST be skipped: their
            // routed_token_ids are stale and would corrupt live token rows.
            // Score scaling already happened in the consumer epilogue (smem).
            //
            // bf16 RMW reduce (vs combine's fp32 accumulate +
            // single bf16 round). Nondeterministic order across topk/split
            // contributors; same class as existing split-K atomics.
            (void)num_splits;
            constexpr int BM = Cfg::MoeDown::BM;
            const int layer_expert = inst.data[gemm_field::LAYER];
            const int layer = layer_expert / NMC_NUM_EXPERTS;
            const int expert = layer_expert % NMC_NUM_EXPERTS;
            const int rows_per_expert = nmc_moe_rows_per_expert(g);
            const int local_row_begin = row_tile * BM - expert * rows_per_expert;
            const int count = g.expert_counts[layer_expert];
            const int valid = max(0, min(BM, count - local_row_begin));
            bf16* x_ffn = reinterpret_cast<bf16*>(g.x_ffn_gl.raw_ptr);
            const bf16* y_base = reinterpret_cast<const bf16*>(&y);
            constexpr uint32_t BYTES = 64 * sizeof(bf16);  // one act_y_tile row
            static_assert(BYTES % 16 == 0, "TMA reduce size must be 16B-aligned");
            for (int r = 0; r < valid; ++r) {
                const int routed = row_tile * BM + r;
                const int token = g.routed_token_ids[nmc_routed_idx(g, layer, routed)];
                bf16* dst = x_ffn + (size_t)token * g.D + col_tile * 64;
                tma_reduce_add_noftz_bf16(
                    dst, y_base + (size_t)r * 64, BYTES);
            }
        } else {
            if (num_splits > 1) tma::store_add_async(g.moe_down_out, y, {0, 0, row_tile, col_tile});
            else                tma::store_async    (g.moe_down_out, y, {0, 0, row_tile, col_tile});
        }
    }
    static __device__ __forceinline__
    void signal_output_bar(const G& g, const NmcInstruction& inst) {
        inst_arrive_out(g.bar_moe_down, inst);
    }
};

template <class Cfg>
__device__ __forceinline__ size_t nmc_topk_idx(const NmcGlobals<Cfg>& g, int layer, int row, int k) {
    return ((size_t)layer * g.BS + row) * NMC_TOPK + k;
}

template <class Cfg>
__device__ __forceinline__ size_t nmc_routed_idx(const NmcGlobals<Cfg>& g, int layer, int routed) {
    return (size_t)layer * g.max_routed + routed;
}

template <class Cfg>
__device__ __forceinline__ int nmc_moe_rows_per_expert(const NmcGlobals<Cfg>& g) {
    constexpr int up_bm = Cfg::MoeUpGate::BM;
    constexpr int down_bm = Cfg::MoeDown::BM;
    constexpr int bm = (up_bm > down_bm) ? up_bm : down_bm;
    const int max_rows = g.BS;
    return ((max_rows + bm - 1) / bm) * bm;
}

template <class Cfg>
__device__ __forceinline__ int nmc_moe_down_bar_idx_from_routed(
    const NmcGlobals<Cfg>& g, int layer, int routed) {
    static_assert(Cfg::MoeDown::BM >= Cfg::MoeUpGate::BM,
        "Fine-grained MoE barriers require Down BM >= UpGate BM");
    static_assert(Cfg::MoeDown::BM % Cfg::MoeUpGate::BM == 0,
        "Fine-grained MoE barriers require Down BM to be a multiple of UpGate BM");
    constexpr int down_bm = Cfg::MoeDown::BM;
    const int rows_per_expert = nmc_moe_rows_per_expert(g);
    const int blocks_per_layer =
        NMC_NUM_EXPERTS * (rows_per_expert / down_bm);
    // `routed` is expert-major within the layer, with rows_per_expert padded
    // to a multiple of down_bm, so integer division directly selects the
    // expert's down-row block.
    return layer * blocks_per_layer + routed / down_bm;
}

template <class Cfg>
__device__ __forceinline__ uint32_t nmc_moe_down_tasks_per_row_block(
    const NmcGlobals<Cfg>& g) {
    constexpr int down_bn = Cfg::MoeDown::BN;
    const int down_n_tiles = (g.D + down_bn - 1) / down_bn;
    return (uint32_t)(Cfg::MoeDown::SPLIT_K * down_n_tiles);
}

// Shared by MOE_COMBINE (baseline) and ADD_RMSNORM (atomic-TMA arm): wait
// until this token's topk down-row-blocks have all arrived on bar_moe_down.
// Caller must already have waited on bar_route (so route_row_for_token is
// valid) and must only call for an active row.
template <class Cfg>
__device__ __forceinline__ void wait_moe_down_for_row(
    const NmcGlobals<Cfg>& g,
    NmcOpSmem<Cfg>& ops,
    int layer,
    int row) {
    #ifdef NMC_MOE_FINE_GRAINED_DOWN_COMBINE
    if (g.bar_moe_down != nullptr) {
        int down_block_indices[NMC_TOPK];
        #pragma unroll
        for (int k = 0; k < NMC_TOPK; ++k) {
            const int routed =
                g.route_row_for_token[nmc_topk_idx(g, layer, row, k)];
            down_block_indices[k] =
                nmc_moe_down_bar_idx_from_routed(g, layer, routed);
        }
        profiled_wait_cross_sm_all_indexed<Cfg>(
            g.bar_moe_down, down_block_indices,
            nmc_moe_down_tasks_per_row_block(g), ops.profiler);
    }
    #else
    (void)row;
    profiled_wait_cross_sm_indexed<Cfg>(
        g.bar_moe_down, layer,
        (uint32_t)g.moe_down_task_count[layer], ops.profiler);
    #endif
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void router_topk_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[router_topk_field::LAYER];
    const int row = inst.data[router_topk_field::ROW];
    const int wait_target = inst.data[3];
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0 && wait_target > 0 && g.bar_router != nullptr) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_router, layer, (uint32_t)wait_target, ops.profiler);
        }
        if (lane_id() == 0) block_profiler_begin<Cfg>(ops.profiler);
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::CONSUMER && CWG_IDX == 0) {
        constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
        const int worker_tid = (int)threadIdx.x - wg_threads;
        static_assert(NMC_NUM_EXPERTS == 128, "router_topk_op warp argmax assumes 128 experts");
        if (worker_tid < kittens::WARP_THREADS) {
            bf16* logits = reinterpret_cast<bf16*>(g.router_logits.raw_ptr);
            float lane_scores[4];
            int lane_experts[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int expert = worker_tid + i * kittens::WARP_THREADS;
                lane_scores[i] = __bfloat162float(logits[(size_t)row * NMC_NUM_EXPERTS + expert]);
                lane_experts[i] = expert;
            }

            if (row_is_active(g, ops.row_active_bits, row)) {
                constexpr uint32_t full_warp = 0xffffffffu;
                // Find winners into per-lane registers, deferring global commits
                // so the top-k atomics and stores can run in parallel.
                int   my_expert = -1;
                float my_score  = 0.0f;
                for (int k = 0; k < NMC_TOPK; ++k) {
                    float best_score = lane_scores[0];
                    int best_expert = lane_experts[0];
                    #pragma unroll
                    for (int i = 1; i < 4; ++i) {
                        const float score = lane_scores[i];
                        const int expert = lane_experts[i];
                        if (score > best_score || (score == best_score && expert < best_expert)) {
                            best_score = score;
                            best_expert = expert;
                        }
                    }

                    #pragma unroll
                    for (int offset = kittens::WARP_THREADS >> 1; offset > 0; offset >>= 1) {
                        const float rhs_score = __shfl_down_sync(full_warp, best_score, offset);
                        const int rhs_expert = __shfl_down_sync(full_warp, best_expert, offset);
                        if (rhs_score > best_score || (rhs_score == best_score && rhs_expert < best_expert)) {
                            best_score = rhs_score;
                            best_expert = rhs_expert;
                        }
                    }

                    // Winner lives in lane 0 after the reduction; broadcast it.
                    const int   selected_expert = __shfl_sync(full_warp, best_expert, 0);
                    const float selected_score  = __shfl_sync(full_warp, best_score, 0);
                    // Lane k owns slot k's commit (k < NMC_TOPK <= warpSize), so
                    // topk[...][k] still holds the k-th best expert as before.
                    if (worker_tid == k) {
                        my_expert = selected_expert;
                        my_score  = selected_score;
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (lane_experts[i] == selected_expert) lane_scores[i] = -INFINITY;
                    }
                }

                // Commit one distinct expert per lane; independent atomic targets
                // make ordering irrelevant and require no additional sync.
                if (worker_tid < NMC_TOPK) {
                    const int rows_per_expert = nmc_moe_rows_per_expert(g);
                    const int slot = atomicAdd(&g.expert_counts[layer * NMC_NUM_EXPERTS + my_expert], 1);
                    const int routed = my_expert * rows_per_expert + slot;
                    const size_t idx = nmc_topk_idx(g, layer, row, worker_tid);
                    const bf16 routed_score = __float2bfloat16(1.0f / (1.0f + __expf(-my_score)));
                    g.topk_experts[idx] = my_expert;
                    g.topk_scores[idx] = routed_score;
                    g.topk_local_slots[idx] = slot;
                    if (slot < rows_per_expert) {
                        g.route_row_for_token[idx] = routed;
                        g.routed_token_ids[nmc_routed_idx(g, layer, routed)] = row;
                        g.routed_scores[nmc_routed_idx(g, layer, routed)] = routed_score;
                    }
                }
            }
            // router_logits is a per-layer scratch buffer reused across MoE
            // layers. Router split-K stores add into it, so clear this row
            // after TOPK has consumed it; the next router layer must not
            // inherit this layer's logits.
            __syncwarp();
            constexpr int VEC_BF16 = sizeof(uint4) / sizeof(bf16);
            if (worker_tid < NMC_NUM_EXPERTS / VEC_BF16) {
                uint4* row_vec = reinterpret_cast<uint4*>(logits + (size_t)row * NMC_NUM_EXPERTS);
                row_vec[worker_tid] = make_uint4(0, 0, 0, 0);
            }
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0) arrive_cross_sm_indexed(g.bar_topk, layer, 1u);
    }
}



template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void route_finalize_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[route_finalize_field::LAYER];
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_topk, layer, (uint32_t)g.BS, ops.profiler);
            block_profiler_begin<Cfg>(ops.profiler);
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::CONSUMER && CWG_IDX == 0) {
        constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
        const int worker_tid = (int)threadIdx.x - wg_threads;
        extern __shared__ int __shm[];
        int* up_prefix = &__shm[0];
        int* down_prefix = up_prefix + NMC_NUM_EXPERTS;
        const int rows_per_expert = nmc_moe_rows_per_expert(g);
        constexpr int up_bn = Cfg::MoeUpGate::BN;
        constexpr int down_bn = Cfg::MoeDown::BN;
        constexpr int up_bm = Cfg::MoeUpGate::BM;
        constexpr int down_bm = Cfg::MoeDown::BM;
        static_assert(down_bm >= up_bm,
            "Fine-grained MoE barriers require Down BM >= UpGate BM");
        static_assert(down_bm % up_bm == 0,
            "Fine-grained MoE barriers require Down BM to be a multiple of UpGate BM");
        const int up_n_tiles = (2 * NMC_EXPERT_DFF + up_bn - 1) / up_bn;
        const int down_n_tiles = (g.D + down_bn - 1) / down_bn;
        const int up_splits = Cfg::MoeUpGate::SPLIT_K;
        const int down_splits = Cfg::MoeDown::SPLIT_K;
        const int e = worker_tid;
        const int count = (e < NMC_NUM_EXPERTS) ? g.expert_counts[layer * NMC_NUM_EXPERTS + e] : 0;
        const int up_rows = (count > 0)
            ? ((count + up_bm - 1) / up_bm)
            : 0;
        const int down_rows = (count > 0)
            ? ((count + down_bm - 1) / down_bm)
            : 0;
        // One consumer lane owns one expert. The shared scans produce stable
        // task offsets; overflow is trapped rather than clipped below.
        if (e < NMC_NUM_EXPERTS) {
            up_prefix[e] = up_rows * up_splits * up_n_tiles;
            down_prefix[e] = down_rows * down_splits * down_n_tiles;
            g.expert_offsets[layer * (NMC_NUM_EXPERTS + 1) + e] = e * rows_per_expert;
        }
        asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(wg_threads) : "memory");

        for (int offset = 1; offset < NMC_NUM_EXPERTS; offset <<= 1) {
            const int up_add = (e >= offset && e < NMC_NUM_EXPERTS) ? up_prefix[e - offset] : 0;
            const int down_add = (e >= offset && e < NMC_NUM_EXPERTS) ? down_prefix[e - offset] : 0;
            asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(wg_threads) : "memory");
            if (e >= offset && e < NMC_NUM_EXPERTS) {
                up_prefix[e] += up_add;
                down_prefix[e] += down_add;
            }
            asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(wg_threads) : "memory");
        }

        const int up_total = up_prefix[NMC_NUM_EXPERTS - 1];
        const int down_total = down_prefix[NMC_NUM_EXPERTS - 1];
        if (e == 0) {
            // Dropless MoE invariant: queue allocation must cover every routed
            // task. This one per-layer guard must remain enabled: otherwise the
            // emission loops below write past the task buffers.
            #if NMC_TRAP_ON_ERROR
            if (up_total > g.max_moe_tasks || down_total > g.max_moe_tasks) {
                __trap();
            }
            #endif
            g.expert_offsets[layer * (NMC_NUM_EXPERTS + 1) + NMC_NUM_EXPERTS] =
                NMC_NUM_EXPERTS * rows_per_expert;
            g.moe_up_task_count[layer] = up_total;
            g.moe_down_task_count[layer] = down_total;
            if (g.moe_up_task_head) g.moe_up_task_head[layer] = 0;
            if (g.moe_down_task_head) g.moe_down_task_head[layer] = 0;
        }

        if (e < NMC_NUM_EXPERTS && count > 0) {
            const int begin = e * rows_per_expert;
            const int layer_expert = layer * NMC_NUM_EXPERTS + e;
            const int up_begin = (e == 0) ? 0 : up_prefix[e - 1];
            const int down_begin = (e == 0) ? 0 : down_prefix[e - 1];
            const int up_end = begin + up_rows * up_bm;
            const int down_end = begin + down_rows * down_bm;
            int local = 0;
            for (int tm = begin / up_bm; tm < up_end / up_bm; ++tm) {
                for (int split = 0; split < up_splits; ++split) {
                    for (int tn = 0; tn < up_n_tiles; ++tn) {
                        const int task = up_begin + local++;
                        #if NMC_TRAP_ON_ERROR
                        if (task >= g.max_moe_tasks) __trap();
                        #endif
                        // Minimal MoE task record, one coalesced int4 store. The
                        // drain scheduler derives opcode, K size, barriers, and
                        // single-tile count from the drain instruction/config;
                        // expert is recomputed from layer_expert. Up tasks don't
                        // use wait_target (store 0 to keep the write vectorized).
                        store_moe_task(g.moe_up_task_words, layer, g.max_moe_tasks, task,
                                       MoeTaskRecord{layer_expert, split, 0, (tm << 16) | tn});
                    }
                }
            }

            local = 0;
            for (int tm = begin / down_bm; tm < down_end / down_bm; ++tm) {
                const int down_block = tm - begin / down_bm;
                const int rows_remaining = count - down_block * down_bm;
                const int rows_in_down = min(rows_remaining, down_bm);
                const int up_blocks_for_down = (rows_in_down + up_bm - 1) / up_bm;
                const int up_wait_target = up_blocks_for_down * up_splits * up_n_tiles;
                for (int split = 0; split < down_splits; ++split) {
                    for (int tn = 0; tn < down_n_tiles; ++tn) {
                        const int task = down_begin + local++;
                        #if NMC_TRAP_ON_ERROR
                        if (task >= g.max_moe_tasks) __trap();
                        #endif
                        store_moe_task(g.moe_down_task_words, layer, g.max_moe_tasks, task,
                                       MoeTaskRecord{layer_expert, split, up_wait_target, (tm << 16) | tn});
                    }
                }
            }
        }
        asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(wg_threads) : "memory");
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0) arrive_cross_sm_indexed(g.bar_route, layer, 1u);
    }
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void moe_gather_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[moe_gather_field::LAYER];
    const int row = inst.data[moe_gather_field::ROW];
    if constexpr (R == NmcRole::PRODUCER) {
        // Gather depends on TOPK's row mapping, not ROUTE_FINALIZE's task words,
        // so it can overlap finalize. Drains still wait for both route and gather.
        if (lane_id() == 0) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_topk, layer, (uint32_t)g.BS, ops.profiler);
            block_profiler_begin<Cfg>(ops.profiler);
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::CONSUMER) {
        bf16* dst_base = reinterpret_cast<bf16*>(g.moe_x.raw_ptr);
        bf16* src_base = reinterpret_cast<bf16*>(g.x_resid.raw_ptr);
        constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
        const int worker_tid = (int)threadIdx.x - wg_threads;
        constexpr int worker_count = 2 * wg_threads;
        constexpr int VEC = 8;
        constexpr int D = 2048;
        static_assert(D % VEC == 0, "NMC hidden size must divide vector width");
        if (row_is_active(g, ops.row_active_bits, row)) {
            for (int k = 0; k < NMC_TOPK; ++k) {
                int routed = g.route_row_for_token[nmc_topk_idx(g, layer, row, k)];
                for (int v = worker_tid; v < D / VEC; v += worker_count) {
                    const int d = v * VEC;
                    uint4 chunk = *reinterpret_cast<const uint4*>(src_base + (size_t)row * g.D + d);
                    *reinterpret_cast<uint4*>(dst_base + (size_t)routed * g.D + d) = chunk;
                }
            }
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0) arrive_cross_sm_indexed(g.bar_gather, layer, 1u);
    }
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void moe_combine_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[moe_combine_field::LAYER];
    const int row = inst.data[moe_combine_field::ROW];
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_route, layer, 1u, ops.profiler);
            if (row_is_active(g, ops.row_active_bits, row)) {
                wait_moe_down_for_row<Cfg>(g, ops, layer, row);
            }
            block_profiler_begin<Cfg>(ops.profiler);
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::CONSUMER) {
        // ADD_RMSNORM waits on one combine arrival per active row.
        bf16* out = reinterpret_cast<bf16*>(g.x_ffn_gl.raw_ptr);
        bf16* expert_out = reinterpret_cast<bf16*>(g.moe_down_out.raw_ptr);
        constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
        const int worker_tid = (int)threadIdx.x - wg_threads;
        constexpr int worker_count = 2 * wg_threads;
        constexpr int VEC = 8;
        constexpr int D = 2048;
        static_assert(D % VEC == 0, "NMC hidden size must divide vector width");
        for (int v = worker_tid; v < D / VEC; v += worker_count) {
            const int d = v * VEC;
            float acc[VEC] = {};
            if (row_is_active(g, ops.row_active_bits, row)) {
                for (int k = 0; k < NMC_TOPK; ++k) {
                    int routed = g.route_row_for_token[nmc_topk_idx(g, layer, row, k)];
                    float score = __bfloat162float(g.routed_scores[nmc_routed_idx(g, layer, routed)]);
                    uint4 chunk = *reinterpret_cast<const uint4*>(expert_out + (size_t)routed * g.D + d);
                    const unsigned* cp = reinterpret_cast<const unsigned*>(&chunk);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 x = __bfloat1622float2(
                            *reinterpret_cast<const __nv_bfloat162*>(cp + j));
                        acc[j * 2] += score * x.x;
                        acc[j * 2 + 1] += score * x.y;
                    }
                }
            }
            uint4 packed;
            unsigned* pp = reinterpret_cast<unsigned*>(&packed);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 y = __floats2bfloat162_rn(acc[j * 2], acc[j * 2 + 1]);
                pp[j] = *reinterpret_cast<const unsigned*>(&y);
            }
            *reinterpret_cast<uint4*>(out + (size_t)row * g.D + d) = packed;
        }
    }
    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0) arrive_cross_sm_indexed(g.bar_ffn_down, layer, 1u);
    }
}

template <NmcRole R, int CWG_IDX, class Cfg, class GemmCfg, class Policy>
__device__ void moe_gemm_drain_op(
    const NmcInstruction& inst,
    const NmcGlobals<Cfg>& g,
    NmcOpSmem<Cfg>& ops,
    int* task_words,
    int* task_count,
    uint32_t* task_head)
{
    const int layer = inst.data[gemm_field::LAYER];
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0 && g.bar_route != nullptr) {
            profiled_wait_cross_sm_indexed<Cfg>(
                g.bar_route, layer, 1u, ops.profiler);
        }
        __syncwarp();
    }
    worker_sync<Cfg>();
    MoeQueueTileScheduler<Cfg, GemmCfg> scheduler(
        inst, task_words, task_count, task_head);
    gemm_op_scheduled<R, CWG_IDX, Cfg, GemmCfg, Policy>(
        inst, g, ops, scheduler);
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void add_rmsnorm_op(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    const int layer = inst.data[rmsnorm_field::LAYER];
    const int row = inst.data[rmsnorm_field::ROW];

    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0) {
            const int tgt_ffn = inst.data[rmsnorm_field::WAIT_FFN_TARGET];
            const int tgt_op = inst.data[rmsnorm_field::WAIT_OPROJ_TARGET];
            // WAIT_FFN_TARGET == -1 is the atomic-TMA MoE-combine sentinel:
            // no MOE_COMBINE arrivals on bar_ffn_down; instead wait on
            // bar_route (route_row_for_token ready) then this row's topk
            // fine-grained bar_moe_down blocks. Dense layers keep tgt_ffn > 0.
            if (tgt_ffn < 0) {
                if constexpr (Cfg::MOE_COMBINE_ATOMIC_TMA) {
                    profiled_wait_cross_sm_indexed<Cfg>(
                        g.bar_route, layer, 1u, ops.profiler);
                    if (row_is_active(g, ops.row_active_bits, row)) {
                        wait_moe_down_for_row<Cfg>(g, ops, layer, row);
                    }
                }
            } else if (tgt_ffn > 0 && g.bar_ffn_down != nullptr) {
                profiled_wait_cross_sm_indexed<Cfg>(
                    g.bar_ffn_down, layer, (uint32_t)tgt_ffn, ops.profiler);
            }
            if (tgt_op > 0 && g.bar_oproj != nullptr) {
                profiled_wait_cross_sm_indexed<Cfg>(
                    g.bar_oproj, layer, (uint32_t)tgt_op, ops.profiler);
            }
            block_profiler_begin<Cfg>(ops.profiler);
        }
        __syncwarp();
    }
    worker_sync<Cfg>();

    if constexpr (R == NmcRole::CONSUMER && CWG_IDX == 0) {
        constexpr int wg_threads = kittens::WARPGROUP_WARPS * kittens::WARP_THREADS;
        // D=2048 gives each thread one 128-bit vector with one consumer WG,
        // minimizing cross-warp reduction and named-barrier participants.
        constexpr int num_consumer_threads = wg_threads;
        constexpr int num_consumer_warps = num_consumer_threads / 32;
        const int tid = (int)threadIdx.x - wg_threads;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const bool active = row_is_active(g, ops.row_active_bits, row);

        extern __shared__ int __shm[];
        float* scratch = reinterpret_cast<float*>(&__shm[0]);

        bf16* x_raw = g.x_raw + (size_t)row * g.D;
        bf16* x_attn = g.x_attn + (size_t)row * g.D;
        bf16* x_ffn = g.x_ffn + (size_t)row * g.D;
        bf16* out = reinterpret_cast<bf16*>(g.x_resid.raw_ptr) + (size_t)row * g.D;
        const bf16* gamma = g.rmsnorm_gamma + (size_t)layer * g.D;

        // NMC RMSNorm is fixed-width D=2048; each consumer thread owns one
        // uint4 containing eight bf16 values. The release model ABI also fixes
        // epsilon at 1e-6 (used in the reduction below).
        constexpr int D = 2048;
        constexpr int VEC = 8;
        constexpr int N_VEC = D / (num_consumer_threads * VEC);
        static_assert(D % (num_consumer_threads * VEC) == 0,
            "NMC RMSNorm width must divide consumer_threads*VEC");

        float vals[VEC * N_VEC];
        float ssq = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_VEC; ++i) {
            const int bidx = (tid + i * num_consumer_threads) * VEC;
            uint4 wr;
            if (active) {
                uint4 vr = *reinterpret_cast<const uint4*>(x_raw + bidx);
                uint4 va = *reinterpret_cast<const uint4*>(x_attn + bidx);
                uint4 vf = *reinterpret_cast<const uint4*>(x_ffn + bidx);
                const unsigned* rp = reinterpret_cast<const unsigned*>(&vr);
                const unsigned* ap = reinterpret_cast<const unsigned*>(&va);
                const unsigned* fp = reinterpret_cast<const unsigned*>(&vf);
                unsigned* wrp = reinterpret_cast<unsigned*>(&wr);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fr = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(rp + j));
                    float2 fa = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(ap + j));
                    float2 ff = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(fp + j));
                    float v0 = fr.x + fa.x + ff.x;
                    float v1 = fr.y + fa.y + ff.y;
                    vals[i * VEC + j * 2] = v0;
                    vals[i * VEC + j * 2 + 1] = v1;
                    ssq += v0 * v0 + v1 * v1;
                    __nv_bfloat162 packed = __floats2bfloat162_rn(v0, v1);
                    wrp[j] = *reinterpret_cast<const unsigned*>(&packed);
                }
                *reinterpret_cast<uint4*>(x_raw + bidx) = wr;
            }
            *reinterpret_cast<uint4*>(x_attn + bidx) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(x_ffn + bidx) = make_uint4(0, 0, 0, 0);
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            ssq += __shfl_xor_sync(0xffffffff, ssq, off);
        }
        if (lane == 0) scratch[warp] = ssq;
        asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(num_consumer_threads) : "memory");

        if (warp == 0) {
            float v = (lane < num_consumer_warps) ? scratch[lane] : 0.0f;
            #pragma unroll
            for (int off = num_consumer_warps / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane == 0) scratch[num_consumer_warps] =
                rsqrtf(v * (1.0f / D) + 1.0e-6f);
        }
        asm volatile("bar.sync %0, %1;\n" :: "r"(6), "r"(num_consumer_threads) : "memory");

        const float rstd = scratch[num_consumer_warps];
        if (active) {
            #pragma unroll
            for (int i = 0; i < N_VEC; ++i) {
                const int bidx = (tid + i * num_consumer_threads) * VEC;
                uint4 vg = *reinterpret_cast<const uint4*>(gamma + bidx);
                const unsigned* gp = reinterpret_cast<const unsigned*>(&vg);
                uint4 vy;
                unsigned* yp = reinterpret_cast<unsigned*>(&vy);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fg = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(gp + j));
                    __nv_bfloat162 y = __floats2bfloat162_rn(
                        vals[i * VEC + j * 2] * rstd * fg.x,
                        vals[i * VEC + j * 2 + 1] * rstd * fg.y);
                    yp[j] = *reinterpret_cast<const unsigned*>(&y);
                }
                *reinterpret_cast<uint4*>(out + bidx) = vy;
            }
        }
        __threadfence();
    }

    worker_sync<Cfg>();
    if constexpr (R == NmcRole::STORER) {
        if (lane_id() == 0 && g.bar_layer != nullptr) {
            arrive_cross_sm_indexed(
                g.bar_layer, inst.data[rmsnorm_field::PRODUCE_BAR_IDX], 1u);
        }
    }
}

__device__ unsigned int layer_sync_counter = 0;
__device__ unsigned int layer_sync_generation = 0;

// For ablation, sync all SMs between waves of instructions.
// Exactly one thread per CTA must call this (the GRID_SYNC op uses producer
// lane 0). Expects gridDim.x == number of participating SMs. Counter/generation
// state persists across launches; a completed sync leaves counter==0 so the
// next launch is safe. A hung mid-sync launch can leave a dirty generation and
// deadlock the next GRID_SYNC until process restart.
__device__ __forceinline__ void grid_sync_barrier() {
    unsigned int num_sms = gridDim.x;
    unsigned int cur_gen = *(volatile unsigned int *)&layer_sync_generation;

    if (atomicAdd(&layer_sync_counter, 1) + 1 == num_sms) {
        layer_sync_counter = 0;
        __threadfence();
        atomicAdd(&layer_sync_generation, 1);
    } else {
        while (*(volatile unsigned int *)&layer_sync_generation == cur_gen) {
            __nanosleep(20);
        }
    }
}

// Ablation GRID_SYNC: one thread per CTA joins the grid barrier; every other
// worker thread idles at worker_sync. Two worker_syncs match the NOP padding
// contract so all roles stay lockstep.
template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void grid_sync_op(
    const NmcInstruction& /*inst*/,
    const NmcGlobals<Cfg>& /*g*/,
    NmcOpSmem<Cfg>& ops) {
    if constexpr (R == NmcRole::PRODUCER) {
        if (lane_id() == 0) {
            block_profiler_begin<Cfg>(ops.profiler);
            grid_sync_barrier();
        }
    }
    worker_sync<Cfg>();
    worker_sync<Cfg>();
}

// Cooperatively load one NMC_INSTRUCTION_WIDTH-int instruction from gmem to smem.
// Called by the controller (warp 0 of WG0) only. `sm_base` is the start of
// this SM's instruction slice (= inst_buf + blockIdx.x * max_inst * IW).
__device__ __forceinline__
void coop_load_inst(NmcInstruction& dst, const int* sm_base, int inst_idx) {
    const int* src = sm_base + inst_idx * NMC_INSTRUCTION_WIDTH;
    int lane = lane_id();
    if (lane < NMC_INSTRUCTION_WIDTH) {
        dst.data[lane] = src[lane];
    }
    __syncwarp();
}

// ── dispatch ─────────────────────────────────────────────────────────────────
// Templated on warp role + cwg_idx so each instantiation contains only the
// code path for that role. ptxas can then size register usage tightly per
// warpgroup (and setmaxnreg is honored).
template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void dispatch_instruction(const NmcInstruction& inst, const NmcGlobals<Cfg>& g, NmcOpSmem<Cfg>& ops) {
    switch (inst.opcode()) {
        case NmcOpcode::FFN_UPGATE_ACT: gemm_op<R, CWG_IDX, Cfg, typename Cfg::UpGate, UpGateActPolicy<Cfg>>(inst, g, ops); break;
        case NmcOpcode::FFN_DOWN:      gemm_op<R, CWG_IDX, Cfg, typename Cfg::Down,   DownPolicy<Cfg>  >(inst, g, ops); break;
        case NmcOpcode::O_PROJ:        gemm_op<R, CWG_IDX, Cfg, typename Cfg::OProj,  OProjPolicy<Cfg> >(inst, g, ops); break;
        case NmcOpcode::LM_HEAD:       gemm_op<R, CWG_IDX, Cfg, typename Cfg::LMHead, LMHeadPolicy<Cfg>>(inst, g, ops); break;
        case NmcOpcode::QKV_PROJ:      qkv_op       <R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ATTN_DECODE:   attn_decode  <R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ATTN_DRAIN:    attn_drain   <R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ATTN_COMBINE:  attn_combine <R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ROUTER_GEMM:   gemm_op<R, CWG_IDX, Cfg, typename Cfg::Router, RouterPolicy<Cfg>>(inst, g, ops); break;
        case NmcOpcode::ROUTER_TOPK:   router_topk_op<R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ROUTE_FINALIZE: route_finalize_op<R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::MOE_GATHER:    moe_gather_op<R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::MOE_UPGATE_ACT_DRAIN:
            moe_gemm_drain_op<R, CWG_IDX, Cfg, typename Cfg::MoeUpGate, MoeUpGateActPolicy<Cfg>>(
                inst, g, ops, g.moe_up_task_words, g.moe_up_task_count, g.moe_up_task_head);
            break;
        case NmcOpcode::MOE_DOWN_DRAIN:
            moe_gemm_drain_op<R, CWG_IDX, Cfg, typename Cfg::MoeDown, MoeDownPolicy<Cfg>>(
                inst, g, ops, g.moe_down_task_words, g.moe_down_task_count, g.moe_down_task_head);
            break;
        case NmcOpcode::MOE_COMBINE:   moe_combine_op<R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::ADD_RMSNORM:   add_rmsnorm_op<R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::GRID_SYNC:     grid_sync_op  <R, CWG_IDX, Cfg>(inst, g, ops); break;
        case NmcOpcode::NOP:
        default: {
            // Even for NOP, idle/wrong-role warps must keep the worker_sync
            // count consistent so the rest of the block doesn't deadlock.
            worker_sync<Cfg>();
            worker_sync<Cfg>();
            break;
        }
    }
}

// ── controller / worker loops ────────────────────────────────────────────────
// Separate top-level functions per warp role. This is the key trick that
// makes ptxas honor `setmaxnreg`: each function's code path is bounded, so
// the per-warpgroup register footprint can diverge.
template <class Cfg>
__device__ void controller_loop(
    const int* sm_base, int num_inst,
    NmcInstruction* inst_ring,
    kittens::semaphore* inst_arrived,
    kittens::semaphore* inst_done)
{
    constexpr int RING = Cfg::INST_RING;
    for (int i = 0; i < num_inst; ++i) {
        int cur = i % RING;
        if (i >= RING) {
            __syncwarp();
            int done_ph = ((i - RING) / RING) & 1;
            kittens::wait(inst_done[cur], done_ph);
            __syncwarp();
        }
        coop_load_inst(inst_ring[cur], sm_base, i);
        __syncwarp();
        if (lane_id() == 0) kittens::arrive(inst_arrived[cur]);
        __syncwarp();
    }
}

template <NmcRole R, int CWG_IDX, class Cfg>
__device__ void worker_loop(
    const NmcGlobals<Cfg>& g, int num_inst,
    NmcInstruction* inst_ring,
    kittens::semaphore* inst_arrived,
    kittens::semaphore* inst_done,
    NmcOpSmem<Cfg>& ops)
{
    constexpr int RING = Cfg::INST_RING;
    for (int i = 0; i < num_inst; ++i) {
        int cur = i % RING;
        __syncwarp();
        auto arrived_ph = (i / RING) % 2;
        kittens::wait(inst_arrived[cur], arrived_ph);
        __syncwarp();

        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) {
                block_profiler_prepare<Cfg>(ops.profiler, inst_ring[cur].opcode());
            }
        }
        dispatch_instruction<R, CWG_IDX, Cfg>(inst_ring[cur], g, ops);

        worker_sync<Cfg>();
        if constexpr (R == NmcRole::PRODUCER) {
            if (lane_id() == 0) block_profiler_finish<Cfg>(ops.profiler);
        }
        if constexpr (R == NmcRole::STORER) {
            if (lane_id() == 0) kittens::arrive(inst_done[cur]);
        }
    }
}

// ── persistent kernel ────────────────────────────────────────────────────────
// grid = #SMs, block = NUM_THREADS.
//   WG0 warp 0   : controller (instruction prefetch)
//   WG0 warp 1   : producer (TMA-loads + sem inits + cross-SM input bar waits)
//   WG0 warp 2   : storer   (TMA-stores + cross-SM output bar arrives)
//   WG0 warp 3   : idle
//   WG1          : consumer (CWG_IDX=0)
//   WG2          : consumer (CWG_IDX=1) when NUM_CWG==2; idle when NUM_CWG==1
template <class Cfg>
__global__ __launch_bounds__(Cfg::NUM_THREADS, 1)
void mk_nmc_kernel(__grid_constant__ const NmcGlobals<Cfg> g) {
    constexpr int RING = Cfg::INST_RING;

    __shared__ NmcInstruction inst_ring[RING];
    __shared__ kittens::semaphore inst_arrived[RING];
    __shared__ kittens::semaphore inst_done[RING];
    __shared__ NmcOpSmem<Cfg> ops;

    if (threadIdx.x == 0) {
        block_profiler_init<Cfg>(ops.profiler, g.prof_buf);
        ops.row_active_bits = load_row_active_bits(g);
        #pragma unroll
        for (int s = 0; s < RING; ++s) {
            kittens::init_semaphore(inst_arrived[s], 0, 1);
            kittens::init_semaphore(inst_done[s],    0, 1);
        }
    }
    __syncthreads();

    const int  bid      = blockIdx.x;
    const int  num_inst = g.num_inst_per_sm[bid];
    const int* sm_base  = g.inst_buf + (int64_t)bid * g.max_inst * NMC_INSTRUCTION_WIDTH;

    const int wg = wg_id();
    if (wg == 0) {
        kittens::warpgroup::decrease_registers<64>();
        const int wiw = warp_in_wg();
        if (wiw == 0)
            controller_loop<Cfg>(sm_base, num_inst, inst_ring, inst_arrived, inst_done);
        else if (wiw == 1)
            worker_loop<NmcRole::PRODUCER, 0, Cfg>(g, num_inst, inst_ring, inst_arrived, inst_done, ops);
        else if (wiw == 2)
            worker_loop<NmcRole::STORER,   0, Cfg>(g, num_inst, inst_ring, inst_arrived, inst_done, ops);
        else
            worker_loop<NmcRole::IDLE,     0, Cfg>(g, num_inst, inst_ring, inst_arrived, inst_done, ops);
    } else if (wg == 1) {
        kittens::warpgroup::increase_registers<216>();
        worker_loop<NmcRole::CONSUMER, 0, Cfg>(
            g, num_inst, inst_ring, inst_arrived, inst_done, ops);
    } else {
        kittens::warpgroup::increase_registers<216>();
        worker_loop<NmcRole::CONSUMER, 1, Cfg>(
            g, num_inst, inst_ring, inst_arrived, inst_done, ops);
    }
}


}  // namespace mk
