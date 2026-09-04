#pragma once
//
// Out-of-tree ThunderKittens extension: WGMMA m64n8k16 (bf16 -> f32).
// ─────────────────────────────────────────────────────────────────────────────
// ThunderKittens only ships base WGMMA tiles for N in {16,32,...,256} (see
// ThunderKittens/include/ops/group/mma/base/64x*.impl). It cannot represent an
// N=8 fragment because its register-tile type (rt / rt_base) is built from 16x16
// atoms. We add the n=8 instruction here WITHOUT editing the TK submodule (it is
// untracked by this repo), using two facts:
//
//   1. `wgmma.mma_async...m64n8k16.f32.bf16.bf16` produces 4 fp32 accumulators
//      per thread (vs 8 for n16).
//   2. Those 4 regs map EXACTLY onto the LEFT 8 columns of a TK float rt_base
//      row-layout tile, i.e. data[0] and data[1]:
//          data[0] = (row = lane/4,     cols { 2*(lane%4), +1 })
//          data[1] = (row = lane/4 + 8, cols { 2*(lane%4), +1 })
//      This is the same c0..c3 register ordering TK's 64x16.impl assigns to
//      data[0..1]; data[2]/data[3] (right 8 cols) are simply left untouched.
//
// We reuse kittens::st_descriptor verbatim, so swizzle / K-stepping is identical
// to the stock TK path; only the instruction's N field (8 vs 16) differs, which
// makes the tensor core read the first 8 N-rows of the B operand instead of 16.
//
// SCOPE / SIMPLIFICATIONS (READ before reusing elsewhere):
//   * ONLY the bf16->f32, shared+shared, A·Bᵀ (trans_a=trans_b=0) form is
//     implemented — the only form the tiny-M GEMM path needs. No rt_st (reg-A),
//     no fp16, no transposed variants.
//   * The production path uses four independent 4-register accumulators, one for
//     each BK=64 K chunk, then sums them in the epilogue. This is intentional:
//     a single 4-register n8 accumulation chain serializes in this kernel.
//   * The B (x) shared tile is physically 16 rows; the n8 instruction consumes
//     only the first 8. The caller MUST guarantee the real token count is <= 8.
//
#include "kittens.cuh"

namespace gemm_n8 {
using namespace kittens;

// A single m64n8k16 writes exactly 4 fp32 registers per thread.
struct alignas(8) acc_n8 {
    float data[4];    // {left.x, left.y, right.x, right.y}
};

// CRITICAL (perf): ptxas serializes a normal n8 accumulation chain in this
// kernel, even with a dedicated 4-register accumulator. The working path keeps
// four independent n8 accumulators, one per BK=64 K chunk, so no accumulator is
// reused inside the same async WGMMA group. The epilogue sums these four partials
// after the final wait.
struct alignas(8) acc_n8x4 {
    acc_n8 part[4];
};

__device__ __forceinline__ void zero_n8(acc_n8& d) {
    d.data[0] = 0.f;
    d.data[1] = 0.f;
    d.data[2] = 0.f;
    d.data[3] = 0.f;
}

__device__ __forceinline__ void zero_n8x4(acc_n8x4& d) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        zero_n8(d.part[i]);
    }
}

__device__ __forceinline__ void mma_fence_n8x4(acc_n8x4& d) {
    asm volatile(
        ""
        : "+f"(d.part[0].data[0]), "+f"(d.part[0].data[1]),
          "+f"(d.part[0].data[2]), "+f"(d.part[0].data[3]),
          "+f"(d.part[1].data[0]), "+f"(d.part[1].data[1]),
          "+f"(d.part[1].data[2]), "+f"(d.part[1].data[3]),
          "+f"(d.part[2].data[0]), "+f"(d.part[2].data[1]),
          "+f"(d.part[2].data[2]), "+f"(d.part[2].data[3]),
          "+f"(d.part[3].data[0]), "+f"(d.part[3].data[1]),
          "+f"(d.part[3].data[2]), "+f"(d.part[3].data[3])
        :: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Four independent accumulators avoid reusing the same accumulator registers
// inside one WGMMA group. Each partial accumulates one of the four BK=64 chunks
// across all K tiles; store_frag_n8x4_sum combines them after the final wait.
__device__ __forceinline__ void wgmma_ss_m64n8k16_bf16_k4_independent_packet(
    acc_n8x4& d,
    uint64_t a0, uint64_t b0,
    uint64_t a1, uint64_t b1,
    uint64_t a2, uint64_t b2,
    uint64_t a3, uint64_t b3) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %16, %17, p, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%4, %5, %6, %7}, %18, %19, p, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%8, %9, %10, %11}, %20, %21, p, 1, 1, 0, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%12, %13, %14, %15}, %22, %23, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d.part[0].data[0]), "+f"(d.part[0].data[1]),
          "+f"(d.part[0].data[2]), "+f"(d.part[0].data[3]),
          "+f"(d.part[1].data[0]), "+f"(d.part[1].data[1]),
          "+f"(d.part[1].data[2]), "+f"(d.part[1].data[3]),
          "+f"(d.part[2].data[0]), "+f"(d.part[2].data[1]),
          "+f"(d.part[2].data[2]), "+f"(d.part[2].data[3]),
          "+f"(d.part[3].data[0]), "+f"(d.part[3].data[1]),
          "+f"(d.part[3].data[2]), "+f"(d.part[3].data[3])
        : "l"(a0), "l"(b0), "l"(a1), "l"(b1),
          "l"(a2), "l"(b2), "l"(a3), "l"(b3));
}

template <ducks::st::all AST, ducks::st::all BST>
__device__ __forceinline__ void mma_ABt_n8_independent(acc_n8x4& d,
                                                       const AST& a,
                                                       const BST& b) {
    static_assert(AST::rows == 64, "A (w) must be 64 rows = wgmma M dim");
    static_assert(std::is_same_v<typename AST::T, bf16> &&
                  std::is_same_v<typename BST::T, bf16>, "bf16 inputs only");
    static_assert(AST::cols == BST::cols, "A and B must share the K dimension");
    constexpr int K = AST::cols / kittens::TILE_COL_DIM<bf16>;
    static_assert(K == 4,
        "independent n8 accumulator path is intentionally scoped to BK=64");
    static_assert(AST::swizzle_bytes == 128 && BST::swizzle_bytes == 128,
        "independent n8 accumulator path assumes 128B-swizzled K-major tiles");

    kittens::st_descriptor<AST, 0> a_desc(a);
    kittens::st_descriptor<BST, 0> b_desc(b);
    mma_fence_n8x4(d);
    wgmma_ss_m64n8k16_bf16_k4_independent_packet(
        d,
        a_desc.chunk_descriptor(0), b_desc.chunk_descriptor(0),
        a_desc.chunk_descriptor(1), b_desc.chunk_descriptor(1),
        a_desc.chunk_descriptor(2), b_desc.chunk_descriptor(2),
        a_desc.chunk_descriptor(3), b_desc.chunk_descriptor(3));
    kittens::warpgroup::mma_commit_group();
}

// Scatter a finished n8 accumulator fragment (64 N-rows x 8 token-cols) into a
// NON-swizzled st_bf<16,64> output tile laid out (token, N) row-major.
//   warp_in_wg : warpgroup::warpid() (0..3) — owns N-rows [16*warp_in_wg, +16)
//   lane       : kittens::laneid()
__device__ __forceinline__ void store_frag_n8(
    bf16* y_ptr, const acc_n8& d, int warp_in_wg, int lane) {
    const int n_lo = warp_in_wg * 16 + lane / 4;   // N index for data[0]
    const int n_hi = n_lo + 8;                      // N index for data[1]
    const int t0   = 2 * (lane % 4);                // token base (cols)
    y_ptr[(t0    ) * 64 + n_lo] = __float2bfloat16(d.data[0]);
    y_ptr[(t0 + 1) * 64 + n_lo] = __float2bfloat16(d.data[1]);
    y_ptr[(t0    ) * 64 + n_hi] = __float2bfloat16(d.data[2]);
    y_ptr[(t0 + 1) * 64 + n_hi] = __float2bfloat16(d.data[3]);
}

__device__ __forceinline__ void store_frag_n8x4_sum(
    bf16* y_ptr, const acc_n8x4& d, int warp_in_wg, int lane) {
    acc_n8 sum;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum.data[i] = d.part[0].data[i] + d.part[1].data[i] +
                      d.part[2].data[i] + d.part[3].data[i];
    }
    store_frag_n8(y_ptr, sum, warp_in_wg, lane);
}

// Like store_frag_n8x4_sum, but multiplies each token-row fragment by a
// per-row scale before the bf16 store. Used by the MOE_COMBINE_ATOMIC_TMA
// arm: s0 scales token row t0, s1 scales t0+1 (see store_frag_n8 layout).
// Scaling happens in fp32 before the single bf16 rounding of the store.
__device__ __forceinline__ void store_frag_n8x4_sum_scaled(
    bf16* y_ptr, const acc_n8x4& d, float s0, float s1,
    int warp_in_wg, int lane) {
    acc_n8 sum;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum.data[i] = d.part[0].data[i] + d.part[1].data[i] +
                      d.part[2].data[i] + d.part[3].data[i];
    }
    sum.data[0] *= s0;
    sum.data[1] *= s1;
    sum.data[2] *= s0;
    sum.data[3] *= s1;
    store_frag_n8(y_ptr, sum, warp_in_wg, lane);
}

__device__ __forceinline__ void store_frag_n8x4_sum_global(
    bf16* y_ptr, int y_cols, const acc_n8x4& d,
    int tile_m, int tile_n, int cwg_idx, int bn, int valid_rows,
    bool add, int warp_in_wg, int lane) {
    acc_n8 sum;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum.data[i] = d.part[0].data[i] + d.part[1].data[i] +
                      d.part[2].data[i] + d.part[3].data[i];
    }

    const int n_lo = cwg_idx * 64 + warp_in_wg * 16 + lane / 4;
    const int n_hi = n_lo + 8;
    const int t0   = 2 * (lane % 4);
    const int row0 = tile_m * 16 + t0;
    const int row1 = row0 + 1;
    const int col0 = tile_n * bn + n_lo;
    const int col1 = tile_n * bn + n_hi;

    if (t0 < valid_rows) {
        bf16* p0 = y_ptr + (size_t)row0 * y_cols + col0;
        bf16* p1 = y_ptr + (size_t)row0 * y_cols + col1;
        if (add) {
            atomicAdd(p0, __float2bfloat16(sum.data[0]));
            atomicAdd(p1, __float2bfloat16(sum.data[2]));
        } else {
            *p0 = __float2bfloat16(sum.data[0]);
            *p1 = __float2bfloat16(sum.data[2]);
        }
    }
    if (t0 + 1 < valid_rows) {
        bf16* p0 = y_ptr + (size_t)row1 * y_cols + col0;
        bf16* p1 = y_ptr + (size_t)row1 * y_cols + col1;
        if (add) {
            atomicAdd(p0, __float2bfloat16(sum.data[1]));
            atomicAdd(p1, __float2bfloat16(sum.data[3]));
        } else {
            *p0 = __float2bfloat16(sum.data[1]);
            *p1 = __float2bfloat16(sum.data[3]);
        }
    }
}

} // namespace gemm_n8
