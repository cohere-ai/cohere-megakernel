"""Single-layer decode correctness: MK Python path vs PyTorch fused-norm baseline.

Loads a real NMC checkpoint once, then sweeps seqlens {128, 1K, 4K, 8K} and
the kernel-supported batch sizes in 1..8 ({1, 2, 4, 8}). Each ``(seqlen, bs)``
prefills independently: MK overwrites the decode-token KV slot, so a max-bs
cache cannot be reused for a later smaller-bs launch.

Each config runs the first dense layer, one SWA MoE layer, and one
full-attention MoE layer independently through the megakernel Python decode
loop. MK never consumes another MK layer's output.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

# ``python src/tests/test_decode_layers.py`` puts src/tests on sys.path[0];
# production modules live one directory up in src/.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import decode.schedule as decode_schedule  # noqa: E402
import native  # noqa: E402
import prefill.torch_prefill as torch_prefill  # noqa: E402


DEFAULT_PROMPT = "Write a short Python function that returns the nth Fibonacci number."
DEFAULT_NUM_SMS = 132
DEFAULT_PAGE_BLOCK = 64
DEFAULT_PREFILL_CHUNK_SIZE = 2048
DEFAULT_VRAM_UTILIZATION = 0.70
DEFAULT_COSINE_MIN = 0.99
DEFAULT_MAX_ABS = 0.50
# 6/8: one or two 8th-place ties are split-K noise; a broken router is not.
DEFAULT_EXPERT_OVERLAP_MIN = 0.75
# Kernel-supported sizes that sit in the requested 1..8 range.
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_SEQLENS = (128, 1024, 4096, 8192)


@dataclass(frozen=True)
class LayerCase:
    """One independently launched decode layer."""

    name: str
    layer_idx: int
    layer_type: str
    is_moe: bool


@dataclass(frozen=True)
class TensorCompare:
    name: str
    cosine: float
    max_abs: float
    mean_abs: float
    rmse: float
    passed: bool


@dataclass
class LayerResult:
    case: LayerCase
    batch: int
    seqlen: int
    compares: list[TensorCompare]
    expert_match: float | None
    passed: bool


class _LayerKvView:
    """Borrow KV pools; 1-layer page-table window over a prefix of rows.

    ``k_pool`` / ``v_pool`` / ``cache_seqlens`` stay views. The page-table
    slice is ``.contiguous()`` copied: ``[layer:layer+1, :bs]`` is packed
    only when ``bs == kv.batch_size``. Copy is safe because the test never
    steps the native handle after this view is built, and the kernel only
    reads the table to index the shared pools.

    ``batch_size`` is the launch batch and must equal ``kv.batch_size``: the
    test prefills each ``(seqlen, bs)`` independently because MK overwrites
    the decode-token KV slot.
    """

    def __init__(
        self,
        kv: torch_prefill.NmcPagedKvCache,
        layer_idx: int,
        batch_size: int,
    ) -> None:
        if batch_size < 1 or batch_size > int(kv.batch_size):
            raise ValueError(
                f"batch_size={batch_size} is outside prefill batch {kv.batch_size}"
            )
        # [layer:layer+1, :bs] is contiguous only when bs == kv.batch_size
        # (layer stride is still full-B). Copy is safe: we never step the KV
        # handle after this view is built, and the kernel only reads the table
        # to index k_pool/v_pool.
        page_table = kv.page_table[layer_idx : layer_idx + 1, :batch_size, :].contiguous()
        self.k_pool = kv.k_pool
        self.v_pool = kv.v_pool
        self.page_table = page_table
        self.cache_seqlens = kv.cache_seqlens[:batch_size]
        self.row_active = kv.row_active[:batch_size]
        self.max_pages = int(kv.max_pages)
        self.page_block = int(kv.page_block)
        self.num_layers = 1
        self.batch_size = int(batch_size)


def _release_library_path(value: str | None) -> str:
    if value is not None:
        return os.path.abspath(value)
    root = os.path.abspath(os.path.join(_SRC_DIR, ".."))
    return os.path.join(root, "build", "libmk_release.so")


def _parse_seqlen_token(value: str) -> int:
    """Accept 128, 1024, 1k, 4K, 8k, etc. ``k`` means *1024."""
    raw = str(value).strip().lower().replace(",", "")
    if raw.endswith("k"):
        return int(float(raw[:-1]) * 1024)
    return int(raw)


def _select_layer_cases(cfg: torch_prefill.NmcConfig) -> list[LayerCase]:
    """First dense, first SWA MoE, first full-attention MoE."""
    dense_idx = 0
    if cfg.first_k_dense_replace < 1:
        raise ValueError("checkpoint has no dense prefix layer to test")
    if cfg.layer_types[dense_idx] != "full_attention":
        raise ValueError("expected layer 0 to be full_attention dense")
    swa_idx = next(
        (
            i
            for i, layer_type in enumerate(cfg.layer_types)
            if layer_type == "sliding_attention" and i >= cfg.first_k_dense_replace
        ),
        None,
    )
    full_moe_idx = next(
        (
            i
            for i, layer_type in enumerate(cfg.layer_types)
            if layer_type == "full_attention" and i >= cfg.first_k_dense_replace
        ),
        None,
    )
    if swa_idx is None or full_moe_idx is None:
        raise ValueError("checkpoint is missing a SWA MoE or full-attention MoE layer")
    return [
        LayerCase(
            name="dense_full",
            layer_idx=dense_idx,
            layer_type=cfg.layer_types[dense_idx],
            is_moe=False,
        ),
        LayerCase(
            name="moe_swa",
            layer_idx=swa_idx,
            layer_type=cfg.layer_types[swa_idx],
            is_moe=True,
        ),
        LayerCase(
            name="moe_full",
            layer_idx=full_moe_idx,
            layer_type=cfg.layer_types[full_moe_idx],
            is_moe=True,
        ),
    ]


def _slice_cfg_for_layer(*, cfg: torch_prefill.NmcConfig, case: LayerCase) -> torch_prefill.NmcConfig:
    """1-layer config whose layer 0 has the target layer's type and MLP kind."""
    # first_k_dense_replace=1 keeps layer 0 dense (and prefix RoPE). 0 makes
    # every layer MoE, so the remapped layer 0 is MoE.
    first_k = 1 if not case.is_moe else 0
    return dataclasses.replace(
        cfg,
        num_hidden_layers=1,
        layer_types=(case.layer_type,),
        first_k_dense_replace=first_k,
    )


def _slice_weights_for_layer(
    *,
    weights: Mapping[str, Any],
    cfg: torch_prefill.NmcConfig,
    layer_idx: int,
) -> dict[str, Any]:
    """Isolate one original layer as a 1-layer MK weight pack.

    ``model.norm.weight`` is replaced with the *next* layer's input LN so the
    1-layer ADD_RMSNORM writes the same destination gamma the full model uses
    after this layer. Embeddings stay shared (make_decode_state still seeds
    from them; the test overwrites ``x_raw`` / ``x_resid`` afterwards).
    """
    num_experts = int(cfg.num_experts)
    moe_start = layer_idx * num_experts
    moe_end = moe_start + num_experts
    if layer_idx + 1 < cfg.num_hidden_layers:
        next_norm = weights["input_layernorm.weight"][layer_idx + 1]
    else:
        next_norm = weights["model.norm.weight"]
    sliced: dict[str, Any] = {
        "W_qkv": weights["W_qkv"][layer_idx : layer_idx + 1],
        "W_oproj": weights["W_oproj"][layer_idx : layer_idx + 1],
        "W_upgate": weights["W_upgate"][layer_idx : layer_idx + 1],
        "W_down": weights["W_down"][layer_idx : layer_idx + 1],
        "W_router": weights["W_router"][layer_idx : layer_idx + 1],
        "W_moe_upgate": weights["W_moe_upgate"][moe_start:moe_end],
        "W_moe_down": weights["W_moe_down"][moe_start:moe_end],
        "input_layernorm.weight": weights["input_layernorm.weight"][layer_idx : layer_idx + 1],
        "model.norm.weight": next_norm,
        "model.embed_tokens.weight": weights["model.embed_tokens.weight"],
        "_upgate_chunk": weights["_upgate_chunk"],
        "_moe_upgate_chunk": weights["_moe_upgate_chunk"],
    }
    # make_decode_state rejects non-contiguous weights. Dim-0 slices of a
    # contiguous stack are usually already packed; copy only if they are not.
    for key, value in list(sliced.items()):
        if isinstance(value, torch.Tensor) and not value.is_contiguous():
            sliced[key] = value.contiguous()
    return sliced


def _compare_tensors(
    *,
    name: str,
    mk_tensor: torch.Tensor,
    pt_tensor: torch.Tensor,
    cosine_min: float,
    max_abs_limit: float,
) -> TensorCompare:
    mk_f = mk_tensor.detach().float().reshape(-1)
    pt_f = pt_tensor.detach().float().reshape(-1)
    if mk_f.numel() != pt_f.numel():
        raise ValueError(
            f"{name} shape mismatch: mk={tuple(mk_tensor.shape)} pt={tuple(pt_tensor.shape)}"
        )
    delta = mk_f - pt_f
    diff = delta.abs()
    mk_norm = torch.linalg.vector_norm(mk_f)
    pt_norm = torch.linalg.vector_norm(pt_f)
    denom = (mk_norm * pt_norm).clamp_min(1.0e-12)
    cosine = float(torch.dot(mk_f, pt_f) / denom)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(delta.square().mean())) if delta.numel() else 0.0
    passed = cosine >= cosine_min and max_abs <= max_abs_limit
    return TensorCompare(
        name=name,
        cosine=cosine,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rmse=rmse,
        passed=passed,
    )


def _expert_overlap_rate(mk_experts: torch.Tensor, pt_experts: torch.Tensor) -> float:
    """Mean per-row |set(mk) ∩ set(pt)| / k.

    All-or-nothing set equality treats a single 8th-place tie as 0.0 even when
    7/8 experts agree. Split-K atomics + bf16 routinely flip that last slot.
    """
    mk_ids = mk_experts.detach().to(torch.int64).cpu()
    pt_ids = pt_experts.detach().to(torch.int64).cpu()
    if mk_ids.shape != pt_ids.shape:
        raise ValueError(
            f"expert shape mismatch: mk={tuple(mk_ids.shape)} pt={tuple(pt_ids.shape)}"
        )
    rows = mk_ids.reshape(-1, mk_ids.shape[-1])
    pt_rows = pt_ids.reshape(-1, pt_ids.shape[-1])
    topk = int(rows.shape[-1])
    if topk < 1:
        raise ValueError("expert tensors have empty top-k axis")
    overlap = 0.0
    for mk_row, pt_row in zip(rows, pt_rows):
        mk_set = set(int(v) for v in mk_row.tolist())
        pt_set = set(int(v) for v in pt_row.tolist())
        overlap += float(len(mk_set & pt_set)) / float(topk)
    return overlap / float(max(1, rows.shape[0]))


def _pytorch_decode_layer(
    *,
    normalized: torch.Tensor,
    raw_residual: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: torch_prefill.NmcConfig,
    flash_attention: Any,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: torch_prefill.NmcPagedKvCache,
    layer_idx: int,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """One decode token through one PyTorch fused-norm layer.

    ``normalized`` / ``raw_residual`` are ``[batch, hidden]``. The prefill
    layer helper already implements decode (seq_len=1) against paged KV via
    ``flash_attn_with_kvcache``. Pages for ``position`` must already be
    allocated (the test does one ``step_decode_positions`` up front).
    """
    batch = int(normalized.shape[0])
    hidden = int(normalized.shape[-1])
    norm_bt = normalized.view(batch, 1, hidden)
    raw_bt = raw_residual.view(batch, 1, hidden)
    router_experts: torch.Tensor | None = None
    router_scores: torch.Tensor | None = None
    if layer_idx >= cfg.first_k_dense_replace:
        _, _, router_scores, router_experts = torch_prefill.compute_moe_routing(
            norm_bt, weights, cfg, layer_idx
        )
    next_norm, next_raw = torch_prefill._transformer_prefill_layer_fused_norms(
        normalized=norm_bt,
        raw_residual=raw_bt,
        weights=weights,
        cfg=cfg,
        flash_attention=flash_attention,
        cos=cos,
        sin=sin,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        position=position,
        trace=None,
    )
    return (
        next_norm.view(batch, hidden),
        next_raw.view(batch, hidden),
        router_experts,
        router_scores,
    )


def _collect_pytorch_layer_io(
    *,
    cases: Sequence[LayerCase],
    weights: Mapping[str, Any],
    cfg: torch_prefill.NmcConfig,
    token: torch.Tensor,
    kv_cache: torch_prefill.NmcPagedKvCache,
    cos: torch.Tensor,
    sin: torch.Tensor,
    flash_attention: Any,
    position: int,
) -> dict[int, dict[str, Any]]:
    """Run PyTorch layers 0..max independently-recorded, sequentially for inputs.

    PyTorch is chained only to produce each tested layer's *input*. MK never
    consumes those outputs. Tested-layer outputs are snapshotted here so the
    later MK launch compares against the same input, not a second PyTorch run
    that would see MK's overwritten decode KV.
    """
    wanted = {case.layer_idx for case in cases}
    max_layer = max(wanted)
    batch = int(token.reshape(-1).shape[0])
    hidden = int(cfg.hidden_size)
    # Keep [B, 1, D]. torch_prefill.rms_norm mark_dynamic's dim 1 as the token
    # axis; a squeezed [B, D] tensor would specialize dim 1 to hidden=2048
    # and trip ConstraintViolationError.
    x_raw = (
        weights["model.embed_tokens.weight"]
        .index_select(0, token.reshape(-1).long())
        .view(batch, 1, hidden)
        .contiguous()
    )
    x_resid = torch_prefill.rms_norm(
        x_raw, weights["input_layernorm.weight"][0], cfg.rms_norm_eps
    ).contiguous()
    captured: dict[int, dict[str, Any]] = {}
    for layer_idx in range(max_layer + 1):
        inp_resid = x_resid[:, 0, :].contiguous()
        inp_raw = x_raw[:, 0, :].contiguous()
        next_norm, next_raw, experts, scores = _pytorch_decode_layer(
            normalized=x_resid[:, 0, :],
            raw_residual=x_raw[:, 0, :],
            weights=weights,
            cfg=cfg,
            flash_attention=flash_attention,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            position=position,
        )
        if layer_idx in wanted:
            captured[layer_idx] = {
                "x_resid_in": inp_resid,
                "x_raw_in": inp_raw,
                "x_resid_out": next_norm.contiguous(),
                "x_raw_out": next_raw.contiguous(),
                "router_experts": None if experts is None else experts.contiguous(),
                "router_scores": None if scores is None else scores.contiguous(),
            }
        x_resid = next_norm.view(batch, 1, hidden)
        x_raw = next_raw.view(batch, 1, hidden)
    return captured


def _run_mk_layer(
    *,
    case: LayerCase,
    full_cfg: torch_prefill.NmcConfig,
    weights: Mapping[str, Any],
    prefill: torch_prefill.NmcReferencePrefillState,
    batch: int,
    x_resid_in: torch.Tensor,
    x_raw_in: torch.Tensor,
    library: Any,
    tiling: decode_schedule.NmcTiling,
    num_sms: int,
    max_attn_splits: int,
    min_attn_chunk: int,
    attn_drain: bool,
    jit_handles: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch one remapped layer through the MK Python decode path."""
    layer_cfg = _slice_cfg_for_layer(cfg=full_cfg, case=case)
    layer_weights = _slice_weights_for_layer(
        weights=weights, cfg=full_cfg, layer_idx=case.layer_idx
    )
    kv_view = _LayerKvView(prefill.kv_cache, case.layer_idx, batch)
    next_token = prefill.next_token.reshape(-1)[:batch]
    hidden = prefill.hidden[:batch]
    slice_prefill = SimpleNamespace(
        input_ids=prefill.input_ids[:batch],
        hidden=hidden,
        next_token=next_token,
        cos=prefill.cos,
        sin=prefill.sin,
        kv_cache=kv_view,
        prompt_len=prefill.prompt_len,
    )
    context_len = int(prefill.prompt_len) + 1
    # Full-attention layers keep production ATTN_DRAIN. Sliding layers never
    # drain; leaving drain on would only build an unused full-attn queue.
    use_drain = bool(attn_drain) and case.layer_type == "full_attention"
    decode_registry = decode_schedule.build_nmc_rr_decode_variant_registry(
        cfg=layer_cfg,
        batch=batch,
        num_sms=num_sms,
        device=prefill.hidden.device,
        tiling=tiling,
        num_layers=1,
        launch_mode="all",
        page_block=int(prefill.kv_cache.page_block),
        max_attn_splits=max_attn_splits,
        min_attn_chunk=min_attn_chunk,
        min_context_len=context_len,
        max_context_len=context_len,
        attn_drain=use_drain,
        attn_drain_sms=None,
        moe_drain_sms=None,
        rr_dependency_affinity=False,
        ablation=False,
    )
    tiling_key = tiling.as_json()
    if tiling_key in jit_handles:
        decode_registry.jit_handles[tiling_key] = jit_handles[tiling_key]
    else:
        decode_registry.ensure_jit_handles(library=library)
        jit_handles[tiling_key] = decode_registry.jit_handle_for(
            decode_registry.variants[0]
        )
    state = decode_schedule.make_decode_state(
        library=library,
        prefill=slice_prefill,
        weights=layer_weights,
        cfg=layer_cfg,
        num_sms=num_sms,
        tiling=tiling,
        decode_registry=decode_registry,
    )
    positions = [
        int(v) for v in kv_view.cache_seqlens[:batch].detach().cpu().tolist()
    ]
    decode_schedule._refresh_attn_drain_queue(state, cache_seqlens=positions)
    decode_schedule._reset(state)
    # 1-layer graph is last-layer: LM waits 2*batch on bar_layer[0], ADD_RMSNORM
    # only arrives batch. Seed the missing batch so the production LM-head wave
    # does not hang. Layer-0 GEMMs use wait_target=0 and do not wait this bar.
    state.tensors["bar_layer"][0] = int(state.desc.BS)
    # Bypass embed+ln0 seed: inject the exact tensors the PyTorch layer saw.
    state.tensors["x_raw"].copy_(x_raw_in)
    state.tensors["x_resid"].copy_(x_resid_in)
    handle = decode_registry.jit_handle_for(state.active_variant)
    decode_schedule._launch_nmc_decode(
        handle=handle,
        state=state,
        device=prefill.hidden.device,
    )
    mk_experts = None
    mk_scores = None
    if case.is_moe:
        mk_experts = state.tensors["topk_experts"][0].clone()
        mk_scores = state.tensors["topk_scores"][0].clone()
    return (
        state.tensors["x_resid"].clone(),
        state.tensors["x_raw"].clone(),
        mk_experts,
        mk_scores,
    )


def _tile_prompt_ids(
    *,
    tokenizer: Any,
    prompt: str,
    seqlen: int,
    batch: int,
    device: torch.device,
    raw_prompt: bool,
) -> torch.Tensor:
    """Broadcast a tiled prompt to ``[batch, seqlen]``.

    Longer seqlens are NOT unique natural text: we repeat the tokenized prompt
    so KV length matches the requested decode context without authoring 8K
    tokens of English.
    """
    if seqlen < 1:
        raise ValueError("seqlen must be positive")
    base = torch_prefill.encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        raw_prompt=raw_prompt,
        add_special_tokens=bool(raw_prompt),
    )
    if int(base.shape[0]) != 1:
        raise ValueError("encode_prompt must return a single row")
    base_len = int(base.shape[1])
    if base_len < 1:
        raise ValueError("prompt tokenization produced no tokens")
    repeats = (seqlen + base_len - 1) // base_len
    tiled = base.repeat(1, repeats)[:, :seqlen].contiguous()
    if batch > 1:
        tiled = tiled.repeat(batch, 1).contiguous()
    return tiled


def _eval_layer(
    *,
    case: LayerCase,
    gold: Mapping[str, Any],
    mk_resid: torch.Tensor,
    mk_raw: torch.Tensor,
    mk_experts: torch.Tensor | None,
    mk_scores: torch.Tensor | None,
    cosine_min: float,
    max_abs_limit: float,
    expert_overlap_min: float,
    batch: int,
    seqlen: int,
) -> LayerResult:
    compares = [
        _compare_tensors(
            name="x_raw",
            mk_tensor=mk_raw,
            pt_tensor=gold["x_raw_out"],
            cosine_min=cosine_min,
            max_abs_limit=max_abs_limit,
        ),
        _compare_tensors(
            name="x_resid",
            mk_tensor=mk_resid,
            pt_tensor=gold["x_resid_out"],
            cosine_min=cosine_min,
            max_abs_limit=max_abs_limit,
        ),
    ]
    expert_match = None
    if case.is_moe:
        if mk_experts is None or gold["router_experts"] is None:
            raise RuntimeError(f"{case.name}: missing MoE top-k tensors")
        expert_match = _expert_overlap_rate(mk_experts, gold["router_experts"])
        compares.append(
            _compare_tensors(
                name="topk_scores",
                mk_tensor=torch.sort(mk_scores.float(), dim=-1).values,
                pt_tensor=torch.sort(gold["router_scores"].float(), dim=-1).values,
                cosine_min=cosine_min,
                max_abs_limit=max_abs_limit,
            )
        )
    passed = all(cmp.passed for cmp in compares)
    if expert_match is not None and expert_match < expert_overlap_min:
        # Collapsed routing (far below a 1-2 expert tie) is a real bug.
        passed = False
        print(
            f"[test] expert overlap {expert_match:.3f} < {expert_overlap_min} "
            f"{case.name} bs={batch} seqlen={seqlen}\n"
            f"  mk_experts={mk_experts.detach().to(torch.int64).cpu().tolist()}\n"
            f"  pt_experts={gold['router_experts'].detach().to(torch.int64).cpu().tolist()}",
            flush=True,
        )
    return LayerResult(
        case=case,
        batch=batch,
        seqlen=seqlen,
        compares=compares,
        expert_match=expert_match,
        passed=passed,
    )


def _print_result(result: LayerResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"=== bs={result.batch} seqlen={result.seqlen} {result.case.name} "
        f"layer={result.case.layer_idx} {result.case.layer_type} "
        f"moe={result.case.is_moe} {status} ===",
        flush=True,
    )
    for cmp in result.compares:
        flag = "ok" if cmp.passed else "BAD"
        print(
            f"  {cmp.name:12s} cosine={cmp.cosine:.6f} rmse={cmp.rmse:.6f} "
            f"max_abs={cmp.max_abs:.5f} mean_abs={cmp.mean_abs:.5f} [{flag}]",
            flush=True,
        )
    if result.expert_match is not None:
        print(f"  experts      overlap={result.expert_match:.3f}", flush=True)


def _print_summary(results: Sequence[LayerResult]) -> None:
    print("[test] summary", flush=True)
    header = (
        f"{'bs':>3} {'seq':>6} {'layer':<12} {'tensor':<12} "
        f"{'cosine':>9} {'rmse':>10} {'max_abs':>9} {'experts':>8} {'ok':>4}"
    )
    print(header, flush=True)
    for result in results:
        expert = "-" if result.expert_match is None else f"{result.expert_match:.3f}"
        for cmp in result.compares:
            flag = "PASS" if cmp.passed else "FAIL"
            print(
                f"{result.batch:3d} {result.seqlen:6d} {result.case.name:<12} "
                f"{cmp.name:<12} {cmp.cosine:9.6f} {cmp.rmse:10.6f} "
                f"{cmp.max_abs:9.5f} {expert:>8} {flag:>4}",
                flush=True,
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="local North Mini Code checkpoint directory",
    )
    parser.add_argument("--lib-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="batch sizes to sweep; default 1 2 4 8 (kernel-supported sizes in 1..8). "
        "3/5/6/7 have no JIT tiling.",
    )
    parser.add_argument(
        "--seqlen",
        type=str,
        nargs="+",
        default=["128", "1k", "4k", "8k"],
        help="prefill lengths to sweep; k=1024. Default: 128 1k 4k 8k",
    )
    parser.add_argument("--num-sms", type=int, default=DEFAULT_NUM_SMS)
    parser.add_argument("--page-block", type=int, default=DEFAULT_PAGE_BLOCK)
    parser.add_argument(
        "--prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE
    )
    parser.add_argument(
        "--frac-vram-utilization", type=float, default=DEFAULT_VRAM_UTILIZATION
    )
    parser.add_argument("--max-attn-splits", type=int, default=None)
    parser.add_argument("--min-attn-chunk", type=int, default=None)
    parser.add_argument(
        "--attn-drain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ATTN_DRAIN on full-attention test layers (production default ON)",
    )
    parser.add_argument("--cosine-min", type=float, default=DEFAULT_COSINE_MIN)
    parser.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument(
        "--expert-overlap-min",
        type=float,
        default=DEFAULT_EXPERT_OVERLAP_MIN,
        help="min mean |intersection|/k for MoE experts (default 0.75 = 6/8)",
    )
    parser.add_argument("--raw-prompt", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    library_path = _release_library_path(args.lib_path)
    if not os.path.isfile(library_path):
        raise FileNotFoundError(
            f"libmk_release.so not found at {library_path}; build with "
            "cmake -S . -B build -G Ninja && cmake --build build"
        )
    native.bind(library_path=library_path, debug=False)
    batch_sizes = tuple(int(v) for v in args.batch_size)
    seqlens = tuple(_parse_seqlen_token(v) for v in args.seqlen)
    unsupported = [b for b in batch_sizes if b not in decode_schedule.SUPPORTED_BATCH_SIZES]
    if unsupported:
        raise ValueError(
            f"batch sizes {unsupported} are not in decode_schedule.SUPPORTED_BATCH_SIZES="
            f"{decode_schedule.SUPPORTED_BATCH_SIZES}. Requested 1..8 maps to 1,2,4,8 "
            "because the TinyM JIT has no tiling for 3/5/6/7."
        )
    if any(s < 1 for s in seqlens):
        raise ValueError("every seqlen must be positive")
    device = torch.device(str(args.device))
    if device.type != "cuda":
        raise ValueError("this test requires a CUDA device")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    max_attn_splits = (
        decode_schedule.DEFAULT_MAX_ATTN_SPLITS
        if args.max_attn_splits is None
        else int(args.max_attn_splits)
    )
    min_attn_chunk = (
        decode_schedule.DEFAULT_MIN_ATTN_CHUNK
        if args.min_attn_chunk is None
        else int(args.min_attn_chunk)
    )

    print(f"[test] loading config checkpoint={args.checkpoint}", flush=True)
    cfg = torch_prefill.NmcConfig.from_checkpoint(str(args.checkpoint))
    cases = _select_layer_cases(cfg)
    print(
        "[test] layers: "
        + ", ".join(f"{c.name}=L{c.layer_idx}({c.layer_type})" for c in cases),
        flush=True,
    )
    print(
        f"[test] sweep seqlens={list(seqlens)} batch_sizes={list(batch_sizes)} "
        "(each pair prefills independently)",
        flush=True,
    )

    # All TinyM (bs<=8) packs share the same UpGate interleave chunk.
    tiling_for_load = decode_schedule.default_nmc_tiling_for_bs(
        batch=1, moe_combine_atomic_tma=False
    )
    upgate_chunk, moe_upgate_chunk = decode_schedule._upgate_weight_chunks_for_tiling(
        tiling_for_load
    )

    print("[test] loading tokenizer", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=False,
        local_files_only=True,
    )
    print("[test] loading MK-layout weights", flush=True)
    weights = torch_prefill.load_weights(
        checkpoint=str(args.checkpoint),
        cfg=cfg,
        device=device,
        dtype=dtype,
        upgate_chunk=upgate_chunk,
        moe_upgate_chunk=moe_upgate_chunk,
    )
    flash_attention = torch_prefill._require_flash_attention()
    library = native.ext()
    tilings = {
        bs: decode_schedule.default_nmc_tiling_for_bs(
            batch=bs, moe_combine_atomic_tma=False
        )
        for bs in batch_sizes
    }
    jit_handles: dict[str, Any] = {}
    results: list[LayerResult] = []
    try:
        for seqlen in seqlens:
            for batch in batch_sizes:
                print(
                    f"[test] pytorch prefill seqlen={seqlen} batch={batch}",
                    flush=True,
                )
                input_ids = _tile_prompt_ids(
                    tokenizer=tokenizer,
                    prompt=str(args.prompt),
                    seqlen=seqlen,
                    batch=batch,
                    device=device,
                    raw_prompt=bool(args.raw_prompt),
                )
                prefill = torch_prefill._nmc_prefill_forward_from_ids(
                    input_ids=input_ids,
                    weights=weights,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                    page_block=int(args.page_block),
                    prefill_chunk_size=int(args.prefill_chunk_size),
                    frac_vram_utilization=float(args.frac_vram_utilization),
                    max_seq=seqlen + 2,
                    temperature=0.0,
                    top_p=1.0,
                    trace=None,
                )
                try:
                    decode_base = [
                        int(v)
                        for v in prefill.kv_cache.cache_seqlens[:batch]
                        .detach()
                        .cpu()
                        .tolist()
                    ]
                    if max(decode_base) != int(prefill.prompt_len):
                        raise RuntimeError(
                            f"prompt_len={prefill.prompt_len} vs "
                            f"max(cache_seqlens)={max(decode_base)}"
                        )
                    if int(prefill.prompt_len) != seqlen:
                        raise RuntimeError(
                            f"tiled prompt_len={prefill.prompt_len} != seqlen={seqlen}"
                        )
                    active = [1] * batch
                    prefill.kv_cache.step_decode_positions(decode_base, active)
                    position = int(decode_base[0])
                    print(
                        f"[test] decode write position={position} seqlen={seqlen} "
                        f"batch={batch}",
                        flush=True,
                    )
                    print(
                        "[test] pytorch per-layer decode (inputs + golden outputs)",
                        flush=True,
                    )
                    captured = _collect_pytorch_layer_io(
                        cases=cases,
                        weights=weights,
                        cfg=cfg,
                        token=prefill.next_token,
                        kv_cache=prefill.kv_cache,
                        cos=prefill.cos,
                        sin=prefill.sin,
                        flash_attention=flash_attention,
                        position=position,
                    )
                    print(
                        f"[test] MK python-loop bs={batch} seqlen={seqlen}",
                        flush=True,
                    )
                    for case in cases:
                        gold = captured[case.layer_idx]
                        mk_resid, mk_raw, mk_experts, mk_scores = _run_mk_layer(
                            case=case,
                            full_cfg=cfg,
                            weights=weights,
                            prefill=prefill,
                            batch=batch,
                            x_resid_in=gold["x_resid_in"],
                            x_raw_in=gold["x_raw_in"],
                            library=library,
                            tiling=tilings[batch],
                            num_sms=int(args.num_sms),
                            max_attn_splits=max_attn_splits,
                            min_attn_chunk=min_attn_chunk,
                            attn_drain=bool(args.attn_drain),
                            jit_handles=jit_handles,
                        )
                        result = _eval_layer(
                            case=case,
                            gold=gold,
                            mk_resid=mk_resid,
                            mk_raw=mk_raw,
                            mk_experts=mk_experts,
                            mk_scores=mk_scores,
                            cosine_min=float(args.cosine_min),
                            max_abs_limit=float(args.max_abs),
                            expert_overlap_min=float(args.expert_overlap_min),
                            batch=batch,
                            seqlen=seqlen,
                        )
                        _print_result(result)
                        results.append(result)
                finally:
                    prefill.close()
                    torch.cuda.empty_cache()
    finally:
        # Native objects that unload on last reference; clearing the dict is
        # the teardown.
        jit_handles.clear()

    _print_summary(results)
    failed = [r for r in results if not r.passed]
    if failed:
        labels = ", ".join(
            f"bs{r.batch}/seq{r.seqlen}/{r.case.name}" for r in failed
        )
        print(f"[test] FAILED: {labels}", flush=True)
        return 1
    print("[test] all configs passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
