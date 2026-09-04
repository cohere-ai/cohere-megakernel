#pragma once

#include "decode/megakernel.cuh"
#include "decode/abi.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef MK_NMC_CUDA_CHECK
#define MK_NMC_CUDA_CHECK(call) do {                                  \
    cudaError_t __e = (call);                                                \
    if (__e != cudaSuccess) {                                                \
        std::fprintf(stderr, "[mk-nmc] %s:%d CUDA %s\n", __FILE__,    \
                __LINE__, cudaGetErrorString(__e));                          \
        std::abort();                                                        \
    }                                                                        \
} while (0)
#endif

namespace mk {

// NmcLaunchDesc and the other POD descriptors now live in decode/abi.h so that the
// nanobind binding TUs can see them without compiling any CUDA.

template <class GL>
inline GL make_nmc_act_gl(bf16* p, unsigned r, unsigned c) {
    return GL(p, nullptr, nullptr, r, c);
}

template <class GL>
inline GL make_nmc_w_layered_gl(bf16* p, unsigned d, unsigned r, unsigned c) {
    return GL(p, nullptr, d, r, c);
}

template <class GL>
inline GL make_nmc_w_lmhead_gl(bf16* p, unsigned r, unsigned c) {
    return GL(p, nullptr, nullptr, r, c);
}

template <class GL>
inline GL make_nmc_kv_gl(bf16* p, unsigned b, unsigned d, unsigned r, unsigned c) {
    return GL(p, b, d, r, c);
}

template <class Cfg>
inline NmcGlobals<Cfg> make_nmc_globals(const NmcLaunchDesc* d) {
    using G = NmcGlobals<Cfg>;

    const unsigned L = (unsigned)d->num_layers;
    const unsigned Bsz = (unsigned)d->BS;
    const unsigned D = (unsigned)d->D;
    const unsigned Dff = (unsigned)d->Dff;
    const unsigned Hq = (unsigned)d->Hq;
    const unsigned Hkv = (unsigned)d->Hkv;
    const unsigned Hd = (unsigned)d->head_dim;
    const unsigned Pg = (unsigned)d->num_phys_pages;
    const unsigned Pb = (unsigned)d->page_block_size;

    const unsigned up_cols = 2u * Dff;
    const unsigned sil_cols = Dff;
    const unsigned q_cols = Hq * Hd;
    const unsigned x_cols = D;
    const unsigned qkv_N = Hq * Hd + 2u * (Hkv * Hd);

    return G{
        reinterpret_cast<int*>(d->inst_buf),
        reinterpret_cast<const int*>(d->num_inst_per_sm),
        d->max_inst,
        make_nmc_w_layered_gl<decltype(G::W_upgate)>((bf16*)d->W_upgate, L, 2u * Dff, D),
        make_nmc_w_layered_gl<decltype(G::W_down)>((bf16*)d->W_down, L, D, Dff),
        make_nmc_w_layered_gl<decltype(G::W_qkv)>((bf16*)d->W_qkv, L, qkv_N, D),
        make_nmc_w_layered_gl<decltype(G::W_oproj)>((bf16*)d->W_oproj, L, D, Hq * Hd),
        make_nmc_w_lmhead_gl<decltype(G::W_lmhead)>((bf16*)d->W_lmhead, 262144u, D),
        make_nmc_w_layered_gl<decltype(G::W_router)>((bf16*)d->W_router, L, NMC_NUM_EXPERTS, D),
        make_nmc_w_layered_gl<decltype(G::W_moe_upgate)>((bf16*)d->W_moe_upgate, L * NMC_NUM_EXPERTS, 2u * NMC_EXPERT_DFF, D),
        make_nmc_w_layered_gl<decltype(G::W_moe_down)>((bf16*)d->W_moe_down, L * NMC_NUM_EXPERTS, D, NMC_EXPERT_DFF),
        make_nmc_kv_gl<decltype(G::K)>((bf16*)d->K_pool, 1u, Pg, Pb, Hkv * Hd),
        make_nmc_kv_gl<decltype(G::V)>((bf16*)d->V_pool, 1u, Pg, Pb, Hkv * Hd),
        (int*)d->page_table,
        (int*)d->cache_seqlens,
        (const int*)d->row_active,
        d->max_pages_per_seq,
        d->page_block_size,
        (const float*)d->cos_table,
        (const float*)d->sin_table,
        (bf16*)d->k_cache,
        (bf16*)d->v_cache,
        make_nmc_act_gl<decltype(G::upgate_scratch)>((bf16*)d->upgate_scratch, Bsz, up_cols),
        make_nmc_act_gl<decltype(G::silu_out)>((bf16*)d->silu_out, Bsz, sil_cols),
        make_nmc_act_gl<decltype(G::q_out)>((bf16*)d->q_out, Bsz, q_cols),
        make_nmc_act_gl<decltype(G::o_proj_in)>((bf16*)d->o_proj_in, Bsz, q_cols),
        make_nmc_act_gl<decltype(G::x_resid)>((bf16*)d->x_resid, Bsz, x_cols),
        make_nmc_act_gl<decltype(G::x_attn_gl)>((bf16*)d->x_attn, Bsz, x_cols),
        make_nmc_act_gl<decltype(G::x_ffn_gl)>((bf16*)d->x_ffn, Bsz, x_cols),
        make_nmc_act_gl<decltype(G::router_logits)>((bf16*)d->router_logits, Bsz, NMC_NUM_EXPERTS),
        make_nmc_act_gl<decltype(G::moe_x)>((bf16*)d->moe_x, (unsigned)d->max_routed, x_cols),
        make_nmc_act_gl<decltype(G::moe_hidden)>((bf16*)d->moe_hidden, (unsigned)d->max_routed, NMC_EXPERT_DFF),
        make_nmc_act_gl<decltype(G::moe_down_out)>((bf16*)d->moe_down_out, (unsigned)d->max_routed, x_cols),
        (bf16*)d->o_partial,
        (float*)d->lse_partial,
        make_nmc_act_gl<decltype(G::lm_logits)>((bf16*)d->lm_logits, Bsz, 262144u),
        (bf16*)d->rmsnorm_gamma,
        (bf16*)d->x_raw,
        (bf16*)d->x_attn,
        (bf16*)d->x_ffn,
        (int*)d->topk_experts,
        (bf16*)d->topk_scores,
        (int*)d->topk_local_slots,
        (int*)d->route_row_for_token,
        (int*)d->routed_token_ids,
        (bf16*)d->routed_scores,
        (int*)d->expert_counts,
        (int*)d->expert_offsets,
        d->max_routed,
        (int*)d->moe_up_task_words,
        (int*)d->moe_down_task_words,
        (int*)d->moe_up_task_count,
        (int*)d->moe_down_task_count,
        (uint32_t*)d->moe_up_task_head,
        (uint32_t*)d->moe_down_task_head,
        d->max_moe_tasks,
        (uint32_t*)d->bar_upgate,
        (uint32_t*)d->bar_silu,
        (uint32_t*)d->bar_ffn_down,
        (uint32_t*)d->bar_qkv,
        (uint32_t*)d->bar_combine,
        (uint32_t*)d->bar_attn,
        (uint32_t*)d->bar_oproj,
        (uint32_t*)d->bar_layer,
        (uint32_t*)d->bar_router,
        (uint32_t*)d->bar_topk,
        (uint32_t*)d->bar_route,
        (uint32_t*)d->bar_gather,
        (uint32_t*)d->bar_moe_upgate,
        (uint32_t*)d->bar_moe_down,
        (uint64_t*)d->prof_buf,
        d->num_layers,
        d->BS,
        d->D,
        d->Dff,
        d->Hq,
        d->Hkv,
        d->num_splits,
        (int*)d->attn_queue_words,
        (uint32_t*)d->attn_queue_heads,
        d->attn_queue_len,
        (int*)d->attn_num_splits,
    };
}

template <class Cfg>
inline void nmc_decode_launch_impl(const NmcLaunchDesc* d) {
    NmcGlobals<Cfg> g = make_nmc_globals<Cfg>(d);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(d->stream_u64);
    // The three-stage attention pipeline raises this kernel's static shared
    // memory above 1 KiB (currently 1,040 bytes). Leave a full 2 KiB margin so
    // static + requested dynamic shared memory stays within the SM90 limit;
    // otherwise cudaFuncSetAttribute fails with cudaErrorInvalidValue.
    int smem_size = kittens::MAX_SHARED_MEMORY - 2048;
    MK_NMC_CUDA_CHECK(cudaFuncSetAttribute(
        mk_nmc_kernel<Cfg>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
    mk_nmc_kernel<Cfg><<<dim3((unsigned)d->num_sms), dim3(Cfg::NUM_THREADS), smem_size, stream>>>(g);
    MK_NMC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace mk
