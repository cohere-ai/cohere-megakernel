"""Standalone prefill and weight-loading module.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.fx.experimental._config as _fx_config
import torch.nn.functional as F
import triton
import triton.language as tl
from kv.pool import (
    NmcPagedKvCache,
    PrefixCacheAttachment,
    allocate_nmc_paged_kv_cache,
)

# Keep dynamic prefill-token and hidden dimensions distinct in Dynamo graphs.
# They are both often 2048, and Dynamo's duck-shape unification would otherwise
# add guards that force a costly recompilation for every final partial chunk.
#
# This is process-global, but this release bundle compiles only the prefill
# helpers below. Do not add unrelated compiled graphs without reconsidering it.
_fx_config.use_duck_shape = False


# Release ABI/layout constants.  Keep these synchronized with kv/cache.h and
# the NMC kernel's fused gate/up tile layout.
DEFAULT_PAGE_BLOCK = 64
# Match the source fused_norms MoE tuning shape used by the release benchmark.
DEFAULT_PREFILL_CHUNK_SIZE = 2048
DEFAULT_UPGATE_LAYOUT_CHUNK = 64
SYNTHETIC_WEIGHT_STDDEV = 0.01
BYTES_PER_GIB = 1024**3
FLASH_ATTENTION_WINDOW_DISABLED = -1
PREFIX_HASH_SEED = b"mk-release-nmc-prefix-v1"


@dataclass(frozen=True)
class NmcConfig:
    """Immutable NMC architecture metadata read from a local checkpoint."""

    hidden_size: int
    intermediate_size: int
    prefix_dense_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    prefix_dense_sliding_window_pattern: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    layer_types: tuple[str, ...]
    eos_token_id: int
    logit_scale: float

    @classmethod
    def release(cls) -> NmcConfig:
        """Baked North Mini Code geometry for synthetic ``--fast`` (no checkpoint).

        ABI fields (hidden size, heads, MoE width, vocab, rms eps) must stay
        synchronized with ``decode/schedule.py`` ``NMC_*`` and
        ``decode/megakernel.cuh``. Layer topology is host-side only: 49 layers,
        full attention every 4th layer, one dense prefix FFN, SWA window 4096.
        """
        num_hidden_layers = 49
        full_attn_period = 4
        layer_types = tuple(
            "full_attention" if (i % full_attn_period) == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        )
        config = cls(
            hidden_size=2048,
            intermediate_size=768,
            prefix_dense_intermediate_size=3072,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=128,
            num_experts_per_tok=8,
            first_k_dense_replace=1,
            prefix_dense_sliding_window_pattern=1,
            vocab_size=262_144,
            rms_norm_eps=1.0e-6,
            rope_theta=50_000.0,
            sliding_window=4096,
            layer_types=layer_types,
            eos_token_id=255_001,
            logit_scale=1.0,
        )
        config.validate()
        return config

    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> NmcConfig:
        """Load only the NMC model configuration from ``checkpoint``."""
        config_path = os.path.join(checkpoint, "config.json")
        with open(config_path, "r", encoding="utf-8") as config_file:
            raw = json.load(config_file)
        config = cls(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            prefix_dense_intermediate_size=int(raw["prefix_dense_intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            num_experts=int(raw["num_experts"]),
            num_experts_per_tok=int(raw["num_experts_per_tok"]),
            first_k_dense_replace=int(raw["first_k_dense_replace"]),
            prefix_dense_sliding_window_pattern=int(
                raw["prefix_dense_sliding_window_pattern"]
            ),
            vocab_size=int(raw["vocab_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            sliding_window=int(raw["sliding_window"]),
            layer_types=tuple(str(value) for value in raw["layer_types"]),
            eos_token_id=int(raw["eos_token_id"]),
            logit_scale=float(raw["logit_scale"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate geometry required by this release prefill implementation."""
        if self.hidden_size <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size and head_dim must be positive")
        if self.num_hidden_layers <= 0 or self.num_attention_heads <= 0:
            raise ValueError("layer and attention-head counts must be positive")
        if self.num_key_value_heads <= 0 or self.num_experts <= 0:
            raise ValueError("key/value head and expert counts must be positive")
        if self.num_experts_per_tok < 1 or self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if self.prefix_dense_sliding_window_pattern < 1:
            raise ValueError("prefix_dense_sliding_window_pattern must be positive")
        # NMC uses a widened Q projection (Hq * head_dim) relative to the
        # residual width. Do not apply the usual dense-transformer equality
        # invariant `hidden_size == num_attention_heads * head_dim` here.


def nmc_kv_sw_pattern(layer_types: Sequence[str]) -> int:
    """MoE/attention hybrid period for the paged-KV manager.

    NMC ``layer_types`` place ``full_attention`` at indices 0, P, 2P, ... and
    ``sliding_attention`` elsewhere. The KV manager's dense test is
    ``(layer % P) == 0``; this helper returns that P.

    Do NOT pass ``prefix_dense_sliding_window_pattern`` as ``sw_pattern`` -- that
    flag only forces RoPE on early dense layers (value 1 = first layer) and is
    not a KV sliding period. Passing it as 1 disables all SWA eviction.
    """
    if not layer_types:
        raise ValueError("layer_types must be non-empty")
    n = len(layer_types)
    for value in layer_types:
        if value not in ("full_attention", "sliding_attention"):
            raise ValueError(f"unknown layer type {value!r}")
    full_idxs = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if not full_idxs:
        # All-sliding is not a NMC topology; the dense test (li % P)==0 always
        # marks layer 0 as dense, so we cannot encode all-sliding with P alone.
        raise ValueError("nmc_kv_sw_pattern requires at least one full_attention layer")
    if len(full_idxs) == 1:
        period = n
    else:
        period = full_idxs[1] - full_idxs[0]
    if period < 1:
        raise ValueError("invalid full_attention spacing in layer_types")
    for i, t in enumerate(layer_types):
        expect_full = (i % period) == 0
        is_full = t == "full_attention"
        if expect_full != is_full:
            raise ValueError(
                f"layer_types[{i}]={t!r} is incompatible with NMC sw_pattern={period} "
                f"(expected {'full_attention' if expect_full else 'sliding_attention'})"
            )
    return int(period)


def max_sliding_window_pages(sw_size: int, page_block: int) -> int:
    """Maximum pages touched by any aligned window of ``sw_size`` tokens.

    A token window need not start on a page boundary. Consequently, a
    4096-token window in a cache with 64-token pages can span 65 pages, not 64.
    This upper bound is used for arena sizing and admission checks.
    """
    if int(page_block) < 1:
        raise ValueError("page_block must be positive")
    if int(sw_size) < 0:
        raise ValueError("sw_size must be non-negative")
    if int(sw_size) == 0:
        return 0
    return (int(sw_size) + 2 * int(page_block) - 2) // int(page_block)


def estimate_hybrid_kv_blocks(
    *,
    layer_types: Sequence[str],
    row_capacities: Sequence[int],
    page_block: int,
    sw_size: int,
) -> int:
    """Physical blocks needed for a hybrid full/SWA cache at the given lengths.

    Full layers keep ``pages(seqlen)`` per row; sliding layers keep at most
    the number of pages touched by an arbitrarily aligned ``sw_size``-token
    window. Summed across rows -- not ``batch * max(seqlen)``.
    """
    if int(page_block) < 1:
        raise ValueError("page_block must be positive")
    if int(sw_size) < 0:
        raise ValueError("sw_size must be non-negative")
    sw_pages = (
        max_sliding_window_pages(int(sw_size), int(page_block))
        if int(sw_size) > 0
        else None
    )
    total = 0
    for layer_type in layer_types:
        sliding = layer_type == "sliding_attention"
        for cap in row_capacities:
            pages = max(1, (max(0, int(cap)) + int(page_block) - 1) // int(page_block))
            if sliding and sw_pages is not None:
                total += min(pages, sw_pages)
            else:
                total += pages
    return int(total)


def hybrid_floor_required_max_seq(
    *,
    cfg: NmcConfig,
    batch: int,
    max_seq: int,
    page_block: int,
    row_capacities: Sequence[int] | None,
    prefill_chunk_size: int,
) -> int:
    """Convert hybrid block demand into ``allocate_nmc_paged_kv_cache``'s floor.

    The allocator still does ``num_layers * batch * pages(required_max_seq)``.
    Express the true hybrid demand as an equivalent uniform per-row length so
    ``required_max_seq=None`` callers (full-context reservation) stop over-
    provisioning as ``batch * max_seq * num_layers``.
    """
    if batch < 1 or max_seq < 1 or page_block < 1:
        raise ValueError("batch, max_seq, and page_block must be positive")
    if prefill_chunk_size < 0:
        raise ValueError("prefill_chunk_size must be non-negative")
    if row_capacities is None:
        caps = [int(max_seq)] * int(batch)
    else:
        caps = [int(v) for v in row_capacities]
        if len(caps) != int(batch):
            raise ValueError(
                f"row_capacities has {len(caps)} rows, expected batch={batch}")
    hybrid_blocks = estimate_hybrid_kv_blocks(
        layer_types=cfg.layer_types,
        row_capacities=caps,
        page_block=int(page_block),
        sw_size=int(cfg.sliding_window),
    )
    # Chunk-outer prefill starts with prepare's end-window resident. Before a
    # chunk is written, each sliding layer can also retain one active history
    # window; writing the whole chunk can add another chunk's pages before the
    # next eviction. Account for that three-region peak, capped by each row's
    # logical prompt capacity. A zero chunk size denotes synthetic KV setup,
    # which never walks the prompt and therefore has no transient peak.
    sw_pages = (
        max_sliding_window_pages(
            int(cfg.sliding_window),
            int(page_block),
        )
        if int(cfg.sliding_window) > 0
        else 0
    )
    chunk_pages = (
        max_sliding_window_pages(int(prefill_chunk_size), int(page_block))
        if int(prefill_chunk_size) > 0
        else 0
    )
    n_sliding = sum(1 for t in cfg.layer_types if t == "sliding_attention")
    for cap in caps:
        prompt_pages = max(
            1,
            (max(0, int(cap)) + int(page_block) - 1) // int(page_block),
        )
        resident_end_pages = min(prompt_pages, sw_pages)
        transient_pages = min(
            max(prompt_pages - resident_end_pages, 0),
            sw_pages + chunk_pages,
        )
        hybrid_blocks += int(n_sliding) * int(transient_pages)
    denom = max(1, int(cfg.num_hidden_layers) * int(batch))
    equiv_pages = (int(hybrid_blocks) + denom - 1) // denom
    return max(1, int(equiv_pages) * int(page_block))


def allocate_paged_kv(
    *,
    cfg: NmcConfig,
    batch: int,
    max_seq: int,
    page_block: int,
    device: torch.device,
    dtype: torch.dtype,
    frac_vram_utilization: float,
    required_max_seq: int | None,
) -> NmcPagedKvCache:
    """Allocate release paged-KV through the single shared native binding.

    The standalone runner receives an ordinary paged cache. The server uses the
    same allocation path, then applies prefix attach/publish in
    ``nmc_prefill_into_slot`` while its decode service is paused.

    ``sw_pattern`` is the hybrid attention period from ``layer_types``, not
    ``prefix_dense_sliding_window_pattern`` (RoPE-only).

    ``required_max_seq=None`` means "floor for a full ``max_seq`` workload":
    use the hybrid full/SWA block count, not ``batch * max_seq * num_layers``.
    An explicit ``required_max_seq`` (session decode reservation) is passed
    through unchanged as a minimum-blocks floor. The implicit floor uses the
    release default chunk size; callers with a different prefill chunk size
    must compute and pass an explicit floor with
    ``hybrid_floor_required_max_seq``.
    """
    floor_seq = (
        hybrid_floor_required_max_seq(
            cfg=cfg,
            batch=int(batch),
            max_seq=int(max_seq),
            page_block=int(page_block),
            row_capacities=None,
            prefill_chunk_size=DEFAULT_PREFILL_CHUNK_SIZE,
        )
        if required_max_seq is None
        else int(required_max_seq)
    )
    return allocate_nmc_paged_kv_cache(
        num_layers=cfg.num_hidden_layers,
        batch_size=batch,
        max_seq=max_seq,
        page_block=page_block,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=dtype,
        device=device,
        sw_size=cfg.sliding_window,
        sw_pattern=nmc_kv_sw_pattern(cfg.layer_types),
        frac_vram_utilization=frac_vram_utilization,
        required_max_seq=floor_seq,
    )


@dataclass
class NmcReferencePrefillState:
    """Caller-owned prefill output consumed by the release MK decoder.

    The state retains CUDA tensors and a borrowed native KV handle. Call
    ``close`` after decode has completed; repeated calls are safe.
    """

    input_ids: torch.Tensor
    hidden: torch.Tensor
    next_token: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    kv_cache: NmcPagedKvCache
    prompt_len: int
    max_seq: int

    def close(self) -> None:
        """Release the paged-KV metadata handle owned by this result."""
        self.kv_cache.close()


@dataclass(frozen=True)
class NmcBatchedPrefillResult:
    """Packed prefill outputs in caller-supplied request order.

    ``next_tokens`` remains on CUDA. The session scatters all first tokens into
    its native generated-token buffer before doing one host synchronization.
    """

    next_tokens: torch.Tensor
    prompt_lengths: tuple[int, ...]
    cached_prefix_lengths: tuple[int, ...]
    computed_tokens: int


class NmcPrefillTrace(Protocol):
    """Debug-only sink for batch-one, single-chunk tensors.

    Besides hidden-width intermediates, traced MoE layers emit router logits,
    top-k scores, and integer expert IDs. Consumers must therefore treat the
    final dimension as stage-specific rather than assuming ``hidden_size``.
    """

    def record(
        self,
        *,
        name: str,
        tensor: torch.Tensor,
        token_start: int,
        token_end: int,
    ) -> None:
        """Persist one named ``[batch, tokens, features]`` intermediate."""


def _prefix_block_hash(parent: bytes, token_ids: Sequence[int]) -> bytes:
    """Hash one full token block, chained to every preceding block."""
    digest = hashlib.sha256()
    digest.update(parent)
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return digest.digest()


def nmc_prefix_block_keys(token_ids: Sequence[int], page_block: int) -> list[bytes]:
    """Return chained SHA-256 keys for cacheable, full prompt blocks.

    The final prompt block is deliberately excluded.  A suffix token must
    always be prefetched after an attach so the correct next-token logits are
    produced from this request's final prompt position.

    Keys intentionally omit model identity because one release process owns one
    checkpoint. Clear the process-global prefix cache before replacing weights
    in a long-lived process.
    """
    if page_block < 1:
        raise ValueError("page_block must be positive")
    parent = PREFIX_HASH_SEED
    keys: list[bytes] = []
    for block_idx in range(max(0, (len(token_ids) - 1) // page_block)):
        start = block_idx * page_block
        parent = _prefix_block_hash(parent, token_ids[start : start + page_block])
        keys.append(parent)
    return keys


def _longest_attachable_prefix_blocks(
    *,
    kv_cache: NmcPagedKvCache,
    keys: Sequence[bytes],
    page_block: int,
    sw_size: int,
) -> int:
    """Return the longest attachable prefix length in blocks.

    A naive scan from ``len(keys)`` downward re-validates the shared stem on
    every miss and goes O(n^2) when the cached stem is slightly shorter than the
    prompt.

    SWA ``required_from_block`` grows with the candidate length, so a *longer*
    prefix can pass when a shorter one fails — binary search on the final
    attach predicate is unsafe. Instead:

    1. Binary-search the longest stem whose *boundary* key still has full-
       attention pages (``required_from_block = block_idx + 1`` makes sliding
       optional). Full-layer retention along the published hash chain is
       treated as monotonic: if ``keys[i]`` is full-ok, ``keys[0:i]`` are too.
       A hole would make ``attach_batch`` fail closed.
    2. Recheck only the SWA-retained tail ``[required_from, stem)`` under the
       real ``required_from_block``. Blocks below that are allowed to lack
       sliding pages and were already accepted for full layers in (1). If the
       tail fails, shrink and recheck tails only (not the whole stem).
    """

    if not keys:
        return 0

    def _full_layer_boundary_ok(block_idx: int) -> bool:
        # required_from = block_idx + 1 => sliding may be absent; full must hit.
        return kv_cache.prefix_can_attach(
            key=keys[block_idx],
            logical_block=block_idx,
            required_from_block=block_idx + 1,
        )

    def _swa_tail_ok(target: int) -> bool:
        required_from_block = max(0, (target * page_block - sw_size) // page_block)
        # Only the live SWA window can still require sliding pages.
        for block_idx in range(target - 1, required_from_block - 1, -1):
            if not kv_cache.prefix_can_attach(
                key=keys[block_idx],
                logical_block=block_idx,
                required_from_block=required_from_block,
            ):
                return False
        return True

    lo = 0
    hi = len(keys)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _full_layer_boundary_ok(mid - 1):
            lo = mid
        else:
            hi = mid - 1
    stem = lo
    if stem < 1:
        return 0

    if _swa_tail_ok(stem):
        return stem
    for target in range(stem - 1, 0, -1):
        if not _full_layer_boundary_ok(target - 1):
            continue
        if _swa_tail_ok(target):
            return target
    return 0


def _attach_prefix_rows(
    kv_cache: NmcPagedKvCache,
    keys_by_row: Sequence[Sequence[bytes]],
    *,
    rows: Sequence[int],
    page_block: int,
    sw_size: int,
) -> tuple[int, ...]:
    """Attach several rows' longest prefixes with one page-table upload."""

    if len(rows) != len(keys_by_row):
        raise ValueError("rows and keys_by_row must have equal lengths")
    attachments: list[PrefixCacheAttachment] = []
    attached_block_counts: list[int] = []
    for row, keys in zip(rows, keys_by_row):
        if row < 0 or row >= kv_cache.batch_size:
            raise ValueError(f"prefix-cache row {row} is out of range")
        attached_blocks = _longest_attachable_prefix_blocks(
            kv_cache=kv_cache,
            keys=keys,
            page_block=page_block,
            sw_size=sw_size,
        )
        attached_block_counts.append(attached_blocks)
        required_from_block = max(
            0, (attached_blocks * page_block - sw_size) // page_block
        )
        attachments.extend(
            PrefixCacheAttachment(
                key=key,
                row=int(row),
                logical_block=block_idx,
                required_from_block=required_from_block,
            )
            for block_idx, key in enumerate(keys[:attached_blocks])
        )

    if attachments and not kv_cache.prefix_attach_batch(attachments):
        raise RuntimeError("validated release prefix-cache batch failed to attach")
    return tuple(count * page_block for count in attached_block_counts)


def _attach_prefix_row(
    kv_cache: NmcPagedKvCache,
    keys: Sequence[bytes],
    *,
    row: int,
    page_block: int,
    sw_size: int,
) -> int:
    """Attach one row through the common single-upload prefix path."""

    return _attach_prefix_rows(
        kv_cache=kv_cache,
        keys_by_row=(keys,),
        rows=(row,),
        page_block=page_block,
        sw_size=sw_size,
    )[0]


def _publish_prefix_row(
    kv_cache: NmcPagedKvCache, keys: Sequence[bytes], *, row: int
) -> int:
    """Publish row-local full prompt blocks for subsequent requests."""
    published = 0
    for block_idx, key in enumerate(keys):
        if kv_cache.prefix_publish(key=key, row=row, logical_block=block_idx):
            published += 1
    return published


class _NmcRowKvView:
    """Expose one live session row to FlashAttention as a batch-one cache."""

    def __init__(self, session_kv: NmcPagedKvCache, row: int) -> None:
        self.k_pool = session_kv.k_pool
        self.v_pool = session_kv.v_pool
        self.page_table = session_kv.page_table[:, row : row + 1, :]
        self.page_block = int(session_kv.page_block)


class _NmcRowsKvView:
    """Expose selected session rows to packed FlashAttention in request order."""

    def __init__(
        self,
        session_kv: NmcPagedKvCache,
        rows: Sequence[int],
    ) -> None:
        if not rows:
            raise ValueError("packed KV view requires at least one row")
        if len(set(rows)) != len(rows):
            raise ValueError("packed KV rows must be distinct")
        if any(row < 0 or row >= session_kv.batch_size for row in rows):
            raise ValueError("packed KV row is out of range")
        self.k_pool = session_kv.k_pool
        self.v_pool = session_kv.v_pool
        first = int(rows[0])
        contiguous = all(int(row) == first + index for index, row in enumerate(rows))
        if contiguous:
            self.page_table = session_kv.page_table[
                :, first : first + len(rows), :
            ]
        else:
            row_indices = torch.tensor(
                rows,
                device=session_kv.page_table.device,
                dtype=torch.long,
            )
            self.page_table = session_kv.page_table.index_select(1, row_indices)
        self.page_block = int(session_kv.page_block)


def _require_flash_attention() -> Any:
    try:
        import flash_attn_interface
    except ImportError as exc:
        raise RuntimeError(
            "paged NMC prefill requires flash_attn_interface from FlashAttention"
        ) from exc
    return flash_attn_interface


def _require_grouped_mm() -> None:
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError(
            "torch._grouped_mm is required for this runner. "
        )


def _rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute RMSNorm in FP32 accumulation and return ``x.dtype``."""
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)


_rms_norm_compiled = torch.compile(_rms_norm_impl)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled standalone RMSNorm used only to seed the first layer."""
    torch._dynamo.mark_dynamic(x, 1)
    return _rms_norm_compiled(x, weight, eps)


def build_rope(max_seq: int, cfg: NmcConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build interleaved GPT-J RoPE cosine/sine tables for absolute positions."""
    if max_seq < 1:
        raise ValueError("max_seq must be positive")
    inv_freq = 1.0 / (
        cfg.rope_theta
        ** (
            torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32)
            / cfg.head_dim
        )
    )
    positions = torch.arange(max_seq, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(frequencies, 2, dim=-1)
    return angles.cos(), angles.sin()


@triton.jit
def _apply_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    tokens,
    heads,
    x_stride_batch,
    x_stride_token,
    x_stride_head,
    x_stride_dim,
    cache_stride_token,
    cache_stride_dim,
    out_stride_batch,
    out_stride_token,
    out_stride_head,
    out_stride_dim,
    HEAD_DIM: tl.constexpr,
    CACHE_AS_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = row % heads
    batch_token = row // heads
    token = batch_token % tokens
    batch = batch_token // tokens

    dim = tl.arange(0, BLOCK_D)
    valid = dim < HEAD_DIM
    peer_dim = dim + tl.where(dim % 2 == 0, 1, -1)
    cache_dim = (dim // 2) * 2

    x_base = (
        batch * x_stride_batch
        + token * x_stride_token
        + head * x_stride_head
    )
    current = tl.load(
        x_ptr + x_base + dim * x_stride_dim,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    peer = tl.load(
        x_ptr + x_base + peer_dim * x_stride_dim,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    cache_offset = (
        token * cache_stride_token + cache_dim * cache_stride_dim
    )
    cos = tl.load(cos_ptr + cache_offset, mask=valid, other=1.0)
    sin = tl.load(sin_ptr + cache_offset, mask=valid, other=0.0)
    # build_rope retains FP32 tables because the native decode kernel consumes
    # float pointers. Match vLLM prefill by explicitly rounding those values to
    # the activation dtype before promoting them back to FP32 for arithmetic.
    if CACHE_AS_BF16:
        cos = cos.to(tl.bfloat16).to(tl.float32)
        sin = sin.to(tl.bfloat16).to(tl.float32)
    else:
        cos = cos.to(tl.float16).to(tl.float32)
        sin = sin.to(tl.float16).to(tl.float32)

    current_cos = current * cos
    peer_sin = peer * sin
    rotated = tl.where(
        dim % 2 == 0,
        current_cos - peer_sin,
        current_cos + peer_sin,
    )
    out_base = (
        batch * out_stride_batch
        + token * out_stride_token
        + head * out_stride_head
    )
    tl.store(
        out_ptr + out_base + dim * out_stride_dim,
        rotated,
        mask=valid,
    )


def apply_rope(
    x: torch.Tensor, cos_slice: torch.Tensor, sin_slice: torch.Tensor
) -> torch.Tensor:
    """Apply interleaved RoPE with explicit vLLM-compatible arithmetic."""
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [batch, tokens, heads, head_dim]")
    batch, tokens, heads, head_dim = (int(size) for size in x.shape)
    if batch < 1 or tokens < 1 or heads < 1:
        raise ValueError("RoPE input dimensions must be positive")
    if head_dim < 2 or head_dim > 256 or head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even and in [2, 256]")
    if tuple(cos_slice.shape) != (tokens, head_dim) or cos_slice.shape != sin_slice.shape:
        raise ValueError("RoPE cache dimensions do not match the input")
    if x.device.type != "cuda" or cos_slice.device != x.device or sin_slice.device != x.device:
        raise ValueError("RoPE input and cache tensors must share one CUDA device")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("RoPE input must use bfloat16 or float16")
    if cos_slice.dtype != torch.float32 or sin_slice.dtype != torch.float32:
        raise ValueError("RoPE cache tensors must use float32")

    output = torch.empty(
        (batch, tokens, heads, head_dim),
        device=x.device,
        dtype=x.dtype,
    )
    block_dim = triton.next_power_of_2(head_dim)
    with torch.cuda.device(x.device):
        _apply_rope_kernel[(batch * tokens * heads,)](
            x,
            cos_slice,
            sin_slice,
            output,
            tokens,
            heads,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            cos_slice.stride(0),
            cos_slice.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            HEAD_DIM=head_dim,
            CACHE_AS_BF16=x.dtype == torch.bfloat16,
            BLOCK_D=block_dim,
            num_warps=4,
        )
    return output


def _copy_interleaved_upgate(
    destination: torch.Tensor, source: torch.Tensor, *, half: int, chunk: int
) -> None:
    """Copy gate/up rows into MK fused-epilogue order.

    The fused epilogue consumes tile-local gate/up halves. The caller supplies
    chunk = fused raw BN / 2 from the active NMC tiling, so this loader does not
    duplicate BN selection policy.
    """
    if source.ndim != 2 or half not in (0, 1) or chunk < 1:
        raise ValueError("invalid interleaved up/gate layout request")
    rows, columns = source.shape
    if rows % chunk != 0:
        raise ValueError("up/gate rows must divide the configured layout chunk")
    destination.view(rows // chunk, 2, chunk, columns)[:, half].copy_(
        source.reshape(rows // chunk, chunk, columns)
    )


def _split_interleaved_upgate_act(
    packed_act: torch.Tensor,
    *,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused up/gate activation into (gate, up) along the feature dim."""
    two_i = int(packed_act.shape[-1])
    inter = two_i // 2
    if inter % chunk != 0:
        raise ValueError(f"intermediate size {inter} must be divisible by chunk {chunk}")
    lead = packed_act.shape[:-1]
    grouped = packed_act.view(*lead, inter // chunk, 2, chunk)
    gate = grouped[..., 0, :].reshape(*lead, inter)
    up = grouped[..., 1, :].reshape(*lead, inter)
    return gate, up


def _weight_layout_chunk(weights: Mapping[str, Any], key: str, fallback: int) -> int:
    value = weights.get(key, fallback)
    return int(value)


def _is_skipped_checkpoint_weight(name: str) -> bool:
    return name.endswith(".bias") or "rotary_emb.inv_freq" in name or name == "lm_head.weight"


@torch.inference_mode()
def load_weights(
    *,
    checkpoint: str,
    cfg: NmcConfig,
    device: torch.device,
    dtype: torch.dtype,
    upgate_chunk: int,
    moe_upgate_chunk: int,
) -> dict[str, Any]:
    """Load local safetensor shards directly into the release NMC weight layout.

    No full checkpoint copy is materialized on the GPU: each CPU shard tensor is
    copied into its final destination and then released with the shard mapping.
    """
    if upgate_chunk < 1 or moe_upgate_chunk < 1:
        raise ValueError("upgate layout chunks must be positive")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("real NMC weight loading requires safetensors") from exc

    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    qkv_width = q_width + 2 * kv_width
    weights: dict[str, Any] = {
        "W_qkv": torch.empty(
            (cfg.num_hidden_layers, qkv_width, cfg.hidden_size), device=device, dtype=dtype
        ),
        "W_oproj": torch.empty(
            (cfg.num_hidden_layers, cfg.hidden_size, q_width), device=device, dtype=dtype
        ),
        "W_upgate": torch.empty(
            (cfg.num_hidden_layers, 2 * cfg.prefix_dense_intermediate_size, cfg.hidden_size),
            device=device,
            dtype=dtype,
        ),
        "W_down": torch.empty(
            (cfg.num_hidden_layers, cfg.hidden_size, cfg.prefix_dense_intermediate_size),
            device=device,
            dtype=dtype,
        ),
        "W_router": torch.empty(
            (cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size), device=device, dtype=dtype
        ),
        "W_moe_upgate": torch.empty(
            (cfg.num_hidden_layers * cfg.num_experts, 2 * cfg.intermediate_size, cfg.hidden_size),
            device=device,
            dtype=dtype,
        ),
        "W_moe_down": torch.empty(
            (cfg.num_hidden_layers * cfg.num_experts, cfg.hidden_size, cfg.intermediate_size),
            device=device,
            dtype=dtype,
        ),
        "input_layernorm.weight": torch.empty(
            (cfg.num_hidden_layers, cfg.hidden_size), device=device, dtype=dtype
        ),
        "model.norm.weight": torch.empty((cfg.hidden_size,), device=device, dtype=dtype),
        "model.embed_tokens.weight": torch.empty(
            (cfg.vocab_size, cfg.hidden_size), device=device, dtype=dtype
        ),
        "_upgate_chunk": int(upgate_chunk),
        "_moe_upgate_chunk": int(moe_upgate_chunk),
    }
    shard_paths = sorted(glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
    if not shard_paths:
        shard_paths = sorted(glob.glob(os.path.join(checkpoint, "worker-*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError("no NMC model-*.safetensors or worker-*.safetensors shards found")
    expert_pattern = re.compile(
        r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)"
        r"\.(?P<projection>gate_proj|up_proj|down_proj)\.weight"
    )
    for shard_path in shard_paths:
        for name, tensor in load_file(shard_path, device="cpu").items():
            if _is_skipped_checkpoint_weight(name):
                continue
            source = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
            if name == "model.embed_tokens.weight" or name == "model.norm.weight":
                weights[name].copy_(source)
            elif name.endswith(".input_layernorm.weight"):
                weights["input_layernorm.weight"][int(name.split(".")[2])].copy_(source)
            elif name.endswith(".self_attn.q_proj.weight"):
                weights["W_qkv"][int(name.split(".")[2]), :q_width].copy_(source)
            elif name.endswith(".self_attn.k_proj.weight"):
                layer = int(name.split(".")[2])
                weights["W_qkv"][layer, q_width : q_width + kv_width].copy_(source)
            elif name.endswith(".self_attn.v_proj.weight"):
                layer = int(name.split(".")[2])
                weights["W_qkv"][layer, q_width + kv_width : qkv_width].copy_(source)
            elif name.endswith(".self_attn.o_proj.weight"):
                weights["W_oproj"][int(name.split(".")[2])].copy_(source)
            elif name.endswith(".mlp.gate.weight"):
                weights["W_router"][int(name.split(".")[2])].copy_(source)
            elif name.endswith(".mlp.gate_proj.weight"):
                _copy_interleaved_upgate(
                    weights["W_upgate"][int(name.split(".")[2])], source, half=0, chunk=upgate_chunk
                )
            elif name.endswith(".mlp.up_proj.weight"):
                _copy_interleaved_upgate(
                    weights["W_upgate"][int(name.split(".")[2])], source, half=1, chunk=upgate_chunk
                )
            elif name.endswith(".mlp.down_proj.weight"):
                weights["W_down"][int(name.split(".")[2])].copy_(source)
            else:
                match = expert_pattern.fullmatch(name)
                if match is None:
                    raise KeyError(f"unhandled NMC checkpoint tensor {name!r}")
                packed = int(match["layer"]) * cfg.num_experts + int(match["expert"])
                projection = match["projection"]
                if projection == "gate_proj":
                    _copy_interleaved_upgate(
                        weights["W_moe_upgate"][packed], source, half=0, chunk=moe_upgate_chunk
                    )
                elif projection == "up_proj":
                    _copy_interleaved_upgate(
                        weights["W_moe_upgate"][packed], source, half=1, chunk=moe_upgate_chunk
                    )
                else:
                    weights["W_moe_down"][packed].copy_(source)
    return weights


def make_fast_nmc_weights(
    *, cfg: NmcConfig, device: torch.device, dtype: torch.dtype, upgate_chunk: int, moe_upgate_chunk: int
) -> dict[str, Any]:
    """Allocate finite random weights for performance-only prefill benchmarking.

    These weights preserve shape and routing traffic, but are not checkpoint
    weights and must never be used as a correctness or quality reference.
    """
    if upgate_chunk < 1 or moe_upgate_chunk < 1:
        raise ValueError("upgate layout chunks must be positive")

    def noise(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype).normal_(0.0, SYNTHETIC_WEIGHT_STDDEV)

    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    return {
        "W_qkv": noise(cfg.num_hidden_layers, q_width + 2 * kv_width, cfg.hidden_size),
        "W_oproj": noise(cfg.num_hidden_layers, cfg.hidden_size, q_width),
        "W_upgate": noise(
            cfg.num_hidden_layers, 2 * cfg.prefix_dense_intermediate_size, cfg.hidden_size
        ),
        "W_down": noise(
            cfg.num_hidden_layers, cfg.hidden_size, cfg.prefix_dense_intermediate_size
        ),
        "W_router": noise(cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size),
        "W_moe_upgate": noise(
            cfg.num_hidden_layers * cfg.num_experts, 2 * cfg.intermediate_size, cfg.hidden_size
        ),
        "W_moe_down": noise(
            cfg.num_hidden_layers * cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
        ),
        "input_layernorm.weight": noise(cfg.num_hidden_layers, cfg.hidden_size),
        "model.norm.weight": noise(cfg.hidden_size),
        "model.embed_tokens.weight": noise(cfg.vocab_size, cfg.hidden_size),
        "_upgate_chunk": int(upgate_chunk),
        "_moe_upgate_chunk": int(moe_upgate_chunk),
    }


def _dense_mlp_impl(hn: torch.Tensor, weights: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    chunk = _weight_layout_chunk(weights, "_upgate_chunk", 64)
    packed_act = hn @ weights["W_upgate"][layer_idx].T
    gate, up = _split_interleaved_upgate_act(packed_act, chunk=chunk)
    return (F.silu(gate) * up) @ weights["W_down"][layer_idx].T


_dense_mlp_compiled = torch.compile(_dense_mlp_impl)


def dense_mlp(hn: torch.Tensor, weights: dict[str, torch.Tensor], layer_idx: int) -> torch.Tensor:
    torch._dynamo.mark_dynamic(hn, 1)
    return _dense_mlp_compiled(hn, weights, layer_idx)


def grouped_expert_mm(
    x: torch.Tensor, weight_out_in: torch.Tensor, group_ends: torch.Tensor
) -> torch.Tensor:
    """Grouped GEMM ``x @ W_e.T`` over resident expert weights."""
    return torch._grouped_mm(x, weight_out_in.transpose(1, 2), group_ends)


@triton.jit
def _bucket_expert_slots_kernel(
    flat_experts_ptr,
    counts_ptr,
    group_ends_ptr,
    expert_counters_ptr,
    inverse_order_ptr,
    sorted_tokens_ptr,
    num_slots,
    top_k,
    BLOCK_SIZE: tl.constexpr,
):
    original_slot = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = original_slot < num_slots
    expert = tl.load(flat_experts_ptr + original_slot, mask=valid, other=0)
    local_slot = tl.atomic_add(expert_counters_ptr + expert, 1, mask=valid)
    expert_start = (
        tl.load(group_ends_ptr + expert, mask=valid, other=0)
        - tl.load(counts_ptr + expert, mask=valid, other=0)
    )
    sorted_slot = expert_start + local_slot
    tl.store(inverse_order_ptr + original_slot, sorted_slot, mask=valid)
    tl.store(sorted_tokens_ptr + sorted_slot, original_slot // top_k, mask=valid)


def bucket_expert_slots(
    flat_experts: torch.Tensor,
    counts: torch.Tensor,
    group_ends: torch.Tensor,
    num_experts_per_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create token-major↔expert-major maps with O(routed slots) work."""
    num_slots = flat_experts.numel()
    expert_counters = torch.zeros_like(group_ends)
    inverse_order = torch.empty_like(flat_experts)
    sorted_tokens = torch.empty_like(flat_experts)
    block_size = 256
    _bucket_expert_slots_kernel[(triton.cdiv(num_slots, block_size),)](
        flat_experts,
        counts,
        group_ends,
        expert_counters,
        inverse_order,
        sorted_tokens,
        num_slots,
        num_experts_per_token,
        BLOCK_SIZE=block_size,
    )
    return inverse_order, sorted_tokens


@triton.jit
def _moe_gather_combine_kernel(
    expert_out_ptr,
    router_scores_ptr,
    inverse_order_ptr,
    out_ptr,
    hidden,
    expert_out_stride,
    score_token_stride,
    score_slot_stride,
    out_token_stride,
    NUM_EXPERTS_PER_TOKEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_hidden = hidden_offset < hidden
    accumulator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for slot in range(NUM_EXPERTS_PER_TOKEN):
        original_slot = token * NUM_EXPERTS_PER_TOKEN + slot
        sorted_slot = tl.load(inverse_order_ptr + original_slot)
        value = tl.load(
            expert_out_ptr + sorted_slot * expert_out_stride + hidden_offset,
            mask=valid_hidden,
            other=0.0,
        ).to(tl.float32)
        score = tl.load(
            router_scores_ptr + token * score_token_stride + slot * score_slot_stride
        ).to(tl.float32)
        accumulator += value * score
    tl.store(out_ptr + token * out_token_stride + hidden_offset, accumulator, mask=valid_hidden)


def moe_gather_combine(
    expert_out: torch.Tensor,
    router_scores: torch.Tensor,
    inverse_order: torch.Tensor,
    num_experts_per_token: int,
) -> torch.Tensor:
    """Gather each token's top-k expert rows and reduce them in FP32."""
    num_tokens = router_scores.shape[0]
    hidden = expert_out.shape[1]
    out = torch.empty((num_tokens, hidden), device=expert_out.device, dtype=expert_out.dtype)
    hidden_block = 1024
    _moe_gather_combine_kernel[(num_tokens, triton.cdiv(hidden, hidden_block))](
        expert_out,
        router_scores,
        inverse_order,
        out,
        hidden,
        expert_out.stride(0),
        router_scores.stride(0),
        router_scores.stride(1),
        out.stride(0),
        NUM_EXPERTS_PER_TOKEN=num_experts_per_token,
        BLOCK_H=hidden_block,
        num_warps=8,
    )
    return out


def compute_moe_routing(
    hn: torch.Tensor, weights: dict[str, torch.Tensor], cfg: NmcConfig, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flattened input, logits, top-k scores, and top-k expert IDs.

    Debug tracing calls this once outside the compiled MoE and the compiled MoE
    calls it again. Keeping one formula prevents the diagnostic path from
    silently drifting from production routing, at the cost of one extra small
    router GEMM only when tracing is enabled.
    """
    batch, seq_len, hidden = hn.shape
    x = hn.reshape(batch * seq_len, hidden)
    router_logits = x @ weights["W_router"][layer_idx].T
    router_scores, router_experts = torch.topk(
        torch.sigmoid(router_logits.float()),
        k=cfg.num_experts_per_tok,
        dim=-1,
        sorted=False,
    )
    return x, router_logits, router_scores, router_experts


def _moe_mlp_impl(
    hn: torch.Tensor, weights: dict[str, torch.Tensor], cfg: NmcConfig, layer_idx: int
) -> torch.Tensor:
    batch, seq_len, hidden = hn.shape
    x, _router_logits, router_scores, router_experts = compute_moe_routing(
        hn, weights, cfg, layer_idx
    )
    flat_experts = router_experts.reshape(-1)
    counts = torch.bincount(flat_experts, minlength=cfg.num_experts)
    group_ends = counts.cumsum(0).to(torch.int32)
    inverse_order, sorted_tokens = bucket_expert_slots(
        flat_experts, counts, group_ends, cfg.num_experts_per_tok
    )
    x_sorted = x.index_select(0, sorted_tokens)
    moe_upgate_layer = weights["W_moe_upgate"].view(
        cfg.num_hidden_layers, cfg.num_experts, 2 * cfg.intermediate_size, cfg.hidden_size
    )[layer_idx]
    moe_down_layer = weights["W_moe_down"].view(
        cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size, cfg.intermediate_size
    )[layer_idx]
    chunk = _weight_layout_chunk(weights, "_moe_upgate_chunk", 64)
    upgate = grouped_expert_mm(x_sorted, moe_upgate_layer, group_ends)
    gate, up = _split_interleaved_upgate_act(upgate, chunk=chunk)
    expert_hidden = F.silu(gate).mul_(up)
    expert_out = grouped_expert_mm(expert_hidden, moe_down_layer, group_ends)
    out = moe_gather_combine(
        expert_out, router_scores.to(x.dtype), inverse_order, cfg.num_experts_per_tok
    )
    return out.view(batch, seq_len, hidden)


_moe_mlp_compiled = torch.compile(_moe_mlp_impl)


def moe_mlp(hn: torch.Tensor, weights: dict[str, torch.Tensor], cfg: NmcConfig, layer_idx: int) -> torch.Tensor:
    torch._dynamo.mark_dynamic(hn, 1)
    return _moe_mlp_compiled(hn, weights, cfg, layer_idx)


def _fused_add_rmsnorm_impl(
    x_raw: torch.Tensor,
    attn_out: torch.Tensor,
    mlp_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_raw = x_raw + attn_out + mlp_out
    x_resid = _rms_norm_impl(x_raw, norm_weight, eps)
    return x_resid, x_raw


_fused_add_rmsnorm_compiled = torch.compile(_fused_add_rmsnorm_impl)


def fused_add_rmsnorm(
    x_raw: torch.Tensor,
    attn_out: torch.Tensor,
    mlp_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._dynamo.mark_dynamic(x_raw, 1)
    torch._dynamo.mark_dynamic(attn_out, 1)
    torch._dynamo.mark_dynamic(mlp_out, 1)
    return _fused_add_rmsnorm_compiled(
        x_raw, attn_out, mlp_out, norm_weight, eps
    )



def _record_attention_trace(
    *,
    trace: NmcPrefillTrace | None,
    stage_prefix: str,
    suffix: str,
    tensor: torch.Tensor,
    batch: int,
    tokens: int,
    position: int,
) -> None:
    """Record one debug-only attention intermediate with heads flattened."""
    if trace is None:
        return
    rows = batch * tokens
    if rows < 1 or tensor.numel() % rows != 0:
        raise ValueError(
            f"attention trace {stage_prefix}_{suffix} cannot be reshaped to "
            f"[{batch}, {tokens}, features] from {tuple(tensor.shape)}"
        )
    trace.record(
        name=f"{stage_prefix}_{suffix}",
        tensor=tensor.reshape(batch, tokens, -1),
        token_start=position,
        token_end=position + tokens,
    )


def _attention_prefill(
    *,
    hidden: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    flash_attention: Any,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: NmcPagedKvCache,
    layer_idx: int,
    position: int,
    attention_trace: NmcPrefillTrace | None,
    attention_trace_prefix: str,
) -> torch.Tensor:
    batch, tokens, _ = hidden.shape
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    qkv = hidden @ weights["W_qkv"][layer_idx].T
    query = qkv[..., :q_width].view(batch, tokens, cfg.num_attention_heads, cfg.head_dim)
    key = qkv[..., q_width : q_width + kv_width].view(
        batch, tokens, cfg.num_key_value_heads, cfg.head_dim
    )
    value = qkv[..., q_width + kv_width :].view(
        batch, tokens, cfg.num_key_value_heads, cfg.head_dim
    )
    for suffix, tensor in (
        ("q_pre_rope", query),
        ("k_pre_rope", key),
        ("value", value),
    ):
        _record_attention_trace(
            trace=attention_trace,
            stage_prefix=attention_trace_prefix,
            suffix=suffix,
            tensor=tensor,
            batch=batch,
            tokens=tokens,
            position=position,
        )
    sliding = cfg.layer_types[layer_idx] == "sliding_attention"
    force_prefix_rope = (
        cfg.first_k_dense_replace > 0
        and cfg.prefix_dense_sliding_window_pattern == 1
        and layer_idx < cfg.first_k_dense_replace
    )
    if sliding or force_prefix_rope:
        query = apply_rope(query, cos[position : position + tokens], sin[position : position + tokens])
        key = apply_rope(key, cos[position : position + tokens], sin[position : position + tokens])
    for suffix, tensor in (("q_post_rope", query), ("k_post_rope", key)):
        _record_attention_trace(
            trace=attention_trace,
            stage_prefix=attention_trace_prefix,
            suffix=suffix,
            tensor=tensor,
            batch=batch,
            tokens=tokens,
            position=position,
        )
    # vLLM interprets a configured SWA size W as W total keys: the current
    # position plus W - 1 prior positions. FlashAttention's left bound is
    # inclusive, so its direct API must receive window_size=(W - 1, 0).
    window_size = (
        (cfg.sliding_window - 1, 0)
        if sliding
        else (FLASH_ATTENTION_WINDOW_DISABLED, FLASH_ATTENTION_WINDOW_DISABLED)
    )
    # Chunk-outer prefill: hand this chunk's K/V to flash_attn_with_kvcache,
    # which appends them into the paged cache at [position, position + tokens)
    # and attends against every cached position [0, position + tokens). Passing
    # cache_seqlens=position (NOT position + tokens) is what lets previously
    # written chunks supply the causal prefix -- the "attention mask adjustment"
    # that makes prefilling one chunk at a time correct.
    # CONTRACT: the caller must have already allocated this chunk's write blocks
    # via kv_cache.step_prefill_chunk(layer_idx, position, position + tokens).
    cache_seqlens = torch.full((batch,), position, device=hidden.device, dtype=torch.int32)
    attended = flash_attention.flash_attn_with_kvcache(
        query,
        kv_cache.k_pool,
        kv_cache.v_pool,
        k=key,
        v=value,
        cache_seqlens=cache_seqlens,
        page_table=kv_cache.page_table[layer_idx],
        causal=True,
        window_size=window_size,
    )
    _record_attention_trace(
        trace=attention_trace,
        stage_prefix=attention_trace_prefix,
        suffix="attended",
        tensor=attended,
        batch=batch,
        tokens=tokens,
        position=position,
    )
    return attended.reshape(batch, tokens, q_width) @ weights["W_oproj"][layer_idx].T


def _transformer_prefill_layer_fused_norms(
    *,
    normalized: torch.Tensor,
    raw_residual: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    flash_attention: Any,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: NmcPagedKvCache,
    layer_idx: int,
    position: int,
    trace: NmcPrefillTrace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One prefill chunk through one layer (MK-style fused Add+RMSNorm)."""
    token_end = position + int(normalized.shape[1])
    stage_prefix = f"layer_{layer_idx:03d}"
    if trace is not None:
        trace.record(
            name=f"{stage_prefix}_input_norm",
            tensor=normalized,
            token_start=position,
            token_end=token_end,
        )
    attention_out = _attention_prefill(
        hidden=normalized,
        weights=weights,
        cfg=cfg,
        flash_attention=flash_attention,
        cos=cos,
        sin=sin,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        position=position,
        attention_trace=None,
        attention_trace_prefix=stage_prefix,
    )
    if trace is not None:
        trace.record(
            name=f"{stage_prefix}_attention_out",
            tensor=attention_out,
            token_start=position,
            token_end=token_end,
        )
    if trace is not None and layer_idx >= cfg.first_k_dense_replace:
        _, router_logits, router_scores, router_experts = compute_moe_routing(
            normalized, weights, cfg, layer_idx
        )
        trace.record(
            name=f"{stage_prefix}_router_logits",
            tensor=router_logits,
            token_start=position,
            token_end=token_end,
        )
        trace.record(
            name=f"{stage_prefix}_router_scores",
            tensor=router_scores,
            token_start=position,
            token_end=token_end,
        )
        trace.record(
            name=f"{stage_prefix}_router_experts",
            tensor=router_experts,
            token_start=position,
            token_end=token_end,
        )
    mlp_out = (
        dense_mlp(normalized, weights, layer_idx)
        if layer_idx < cfg.first_k_dense_replace
        else moe_mlp(normalized, weights, cfg, layer_idx)
    )
    if trace is not None:
        trace.record(
            name=f"{stage_prefix}_mlp_out",
            tensor=mlp_out,
            token_start=position,
            token_end=token_end,
        )
    norm_weight = (
        weights["input_layernorm.weight"][layer_idx + 1]
        if layer_idx + 1 < cfg.num_hidden_layers
        else weights["model.norm.weight"]
    )
    next_normalized, next_raw = fused_add_rmsnorm(
        raw_residual, attention_out, mlp_out, norm_weight, cfg.rms_norm_eps
    )
    if trace is not None:
        trace.record(
            name=f"{stage_prefix}_output_raw",
            tensor=next_raw,
            token_start=position,
            token_end=token_end,
        )
    return next_normalized, next_raw


def _attention_prefill_packed(
    *,
    hidden: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    flash_attention: Any,
    packed_cos: torch.Tensor,
    packed_sin: torch.Tensor,
    kv_cache: _NmcRowsKvView,
    layer_idx: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Run one paged varlen attention call over concatenated real suffixes."""

    _, total_tokens, _ = hidden.shape
    q_width = cfg.num_attention_heads * cfg.head_dim
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    qkv = hidden @ weights["W_qkv"][layer_idx].T
    query = qkv[..., :q_width].view(
        1, total_tokens, cfg.num_attention_heads, cfg.head_dim
    )
    key = qkv[..., q_width : q_width + kv_width].view(
        1, total_tokens, cfg.num_key_value_heads, cfg.head_dim
    )
    value = qkv[..., q_width + kv_width :].view(
        total_tokens, cfg.num_key_value_heads, cfg.head_dim
    )
    sliding = cfg.layer_types[layer_idx] == "sliding_attention"
    force_prefix_rope = (
        cfg.first_k_dense_replace > 0
        and cfg.prefix_dense_sliding_window_pattern == 1
        and layer_idx < cfg.first_k_dense_replace
    )
    if sliding or force_prefix_rope:
        query = apply_rope(query, packed_cos, packed_sin)
        key = apply_rope(key, packed_cos, packed_sin)
    query = query.view(total_tokens, cfg.num_attention_heads, cfg.head_dim)
    key = key.view(total_tokens, cfg.num_key_value_heads, cfg.head_dim)
    window_size = (
        (cfg.sliding_window - 1, 0)
        if sliding
        else (FLASH_ATTENTION_WINDOW_DISABLED, FLASH_ATTENTION_WINDOW_DISABLED)
    )
    attended = flash_attention.flash_attn_with_kvcache(
        q=query,
        k_cache=kv_cache.k_pool,
        v_cache=kv_cache.v_pool,
        k=key,
        v=value,
        cache_seqlens=cache_seqlens,
        page_table=kv_cache.page_table[layer_idx],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k_new=cu_seqlens,
        max_seqlen_q=max_seqlen,
        causal=True,
        window_size=window_size,
    )
    if isinstance(attended, tuple):
        attended = attended[0]
    return (
        attended.reshape(1, total_tokens, q_width)
        @ weights["W_oproj"][layer_idx].T
    )


def _transformer_prefill_layer_packed(
    *,
    normalized: torch.Tensor,
    raw_residual: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    flash_attention: Any,
    packed_cos: torch.Tensor,
    packed_sin: torch.Tensor,
    kv_cache: _NmcRowsKvView,
    layer_idx: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One packed suffix microbatch through one fused-norm NMC layer."""

    attention_out = _attention_prefill_packed(
        hidden=normalized,
        weights=weights,
        cfg=cfg,
        flash_attention=flash_attention,
        packed_cos=packed_cos,
        packed_sin=packed_sin,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        cache_seqlens=cache_seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    mlp_out = (
        dense_mlp(normalized, weights, layer_idx)
        if layer_idx < cfg.first_k_dense_replace
        else moe_mlp(normalized, weights, cfg, layer_idx)
    )
    norm_weight = (
        weights["input_layernorm.weight"][layer_idx + 1]
        if layer_idx + 1 < cfg.num_hidden_layers
        else weights["model.norm.weight"]
    )
    return fused_add_rmsnorm(
        raw_residual,
        attention_out,
        mlp_out,
        norm_weight,
        cfg.rms_norm_eps,
    )


def _validate_sampling_params(temperature: float, top_p: float) -> None:
    if (
        not math.isfinite(float(temperature))
        or float(temperature) < 0.0
        or not math.isfinite(float(top_p))
        or not 0.0 < float(top_p) <= 1.0
    ):
        raise ValueError(
            "temperature must be finite and non-negative; "
            "top_p must be finite and in (0, 1]"
        )


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    _validate_sampling_params(temperature, top_p)
    temperature = float(temperature)
    top_p = float(top_p)
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Keep the first token whose cumulative mass crosses top_p. Filtering
        # directly with ``cumulative <= top_p`` can retain less than top_p mass.
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)


def _sample_next_tokens(
    logits: torch.Tensor,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
) -> torch.Tensor:
    """Sample heterogeneous rows while keeping the common policy batched."""

    if logits.ndim != 2 or len(temperatures) != int(logits.shape[0]):
        raise ValueError("logits and temperatures must have matching row counts")
    if len(top_ps) != int(logits.shape[0]):
        raise ValueError("logits and top_ps must have matching row counts")
    if all(
        float(value) == float(temperatures[0])
        for value in temperatures
    ) and all(float(value) == float(top_ps[0]) for value in top_ps):
        return _sample_next_token(
            logits,
            temperature=float(temperatures[0]),
            top_p=float(top_ps[0]),
        )
    return torch.cat(
        [
            _sample_next_token(
                logits[index : index + 1],
                temperature=float(temperature),
                top_p=float(top_p),
            )
            for index, (temperature, top_p) in enumerate(
                zip(temperatures, top_ps)
            )
        ],
        dim=0,
    )


@torch.inference_mode()
def _nmc_prefill_forward_from_ids(
    *,
    input_ids: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    device: torch.device,
    dtype: torch.dtype,
    page_block: int,
    prefill_chunk_size: int,
    frac_vram_utilization: float,
    max_seq: int,
    temperature: float,
    top_p: float,
    trace: NmcPrefillTrace | None,
    row_lengths: Sequence[int] | None = None,
) -> NmcReferencePrefillState:
    """Prefill already-tokenized NMC inputs into the release paged KV cache.

    ``row_lengths`` enables RAGGED prefill: ``input_ids`` is right-padded to a
    rectangular ``(batch, max_len)`` and every row is computed over the full
    width, but only the first ``row_lengths[r]`` tokens of row ``r`` are real.
    Because attention is causal, padding after a row's real tokens never
    affects those tokens' KV/hidden. After the forward we (a) set each row's
    ``cache_seqlens`` to its true length so decode attends only real KV, and
    (b) sample ``next_token`` from each row's last real position. 
    ragged prefill requires a single chunk (``prefill_chunk_size >= max_len``)
    so the full per-row residual stream is available for that gather.

    Debug tracing has the same explicit single-chunk restriction and additionally
    requires batch one. This keeps every dumped stage directly comparable with
    one eager vLLM prefill tensor; normal untraced prefill remains chunkable.
    """
    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1 or int(input_ids.shape[1]) < 1:
        raise ValueError("input_ids must have shape [batch >= 1, prompt_len >= 1]")
    # `torch.device("cuda")` denotes the current device, while tensors report
    # the concrete ordinal (`cuda:0`). Treat those forms as equivalent.
    if input_ids.device.type != device.type or (
        device.index is not None and input_ids.device.index != device.index
    ):
        raise ValueError("input_ids must already be on device")
    if prefill_chunk_size < 1:
        raise ValueError("prefill_chunk_size must be positive")
    batch = int(input_ids.shape[0])
    prompt_len = int(input_ids.shape[1])
    if max_seq < prompt_len:
        raise ValueError("max_seq must include the complete prompt")
    if trace is not None:
        if batch != 1:
            raise ValueError("prefill hidden-state tracing requires batch=1")
        if prefill_chunk_size < prompt_len:
            raise ValueError(
                "prefill hidden-state tracing requires one complete chunk "
                f"(prefill_chunk_size >= prompt_len={prompt_len})"
            )
    resolved_row_lengths: list[int] | None = None
    if row_lengths is not None:
        resolved_row_lengths = [int(v) for v in row_lengths]
        if len(resolved_row_lengths) != batch:
            raise ValueError(
                f"row_lengths length {len(resolved_row_lengths)} must equal batch {batch}")
        if min(resolved_row_lengths) < 1:
            raise ValueError("every ragged prompt must tokenize to >= 1 token")
        if max(resolved_row_lengths) != prompt_len:
            raise ValueError(
                f"ragged prefill expects input_ids padded to max(row_lengths)="
                f"{max(resolved_row_lengths)}, got width {prompt_len}")
        if prompt_len > prefill_chunk_size:
            raise ValueError(
                "ragged prefill requires a single chunk (prefill_chunk_size >= "
                f"max_len={prompt_len}); got prefill_chunk_size={prefill_chunk_size}")
    flash_attention = _require_flash_attention()
    _require_grouped_mm()
    cos, sin = build_rope(max_seq, cfg, device)
    kv_floor_seq = hybrid_floor_required_max_seq(
        cfg=cfg,
        batch=int(batch),
        max_seq=int(max_seq),
        page_block=int(page_block),
        row_capacities=None,
        prefill_chunk_size=int(prefill_chunk_size),
    )
    kv_cache = allocate_paged_kv(
        cfg=cfg,
        batch=batch,
        max_seq=max_seq,
        page_block=page_block,
        device=device,
        dtype=dtype,
        frac_vram_utilization=frac_vram_utilization,
        required_max_seq=kv_floor_seq,
    )
    try:
        # Pages are reserved for the rectangular padded width; true per-row
        # lengths are published via set_cache_seqlens after the forward.
        kv_cache.prepare_prefill_lengths([prompt_len] * batch)
        # Chunk-outer prefill: embed one chunk of the prompt and push it through
        # ALL layers before moving to the next chunk. Only the current chunk's
        # residual stream (B, chunk, H) is materialized at a time; prior chunks
        # live solely in the paged KV cache, which the attention path attends
        # against for causal context. This bounds residual-stream activation
        # memory to O(B * chunk * H) instead of O(B * prompt_len * H).
        #
        # INTENTIONAL TRADEOFF (memory, not throughput): every layer's weights
        # are re-read once per chunk with no cross-chunk reuse, so this is slower
        # than a single-pass layer-outer prefill for prompts that fit in memory.
        # When prompt_len <= prefill_chunk_size there is exactly one chunk, which
        # degenerates to the whole-prompt prefill.
        final_hidden: torch.Tensor | None = None
        embedding = weights["model.embed_tokens.weight"]
        for chunk_start in range(0, prompt_len, prefill_chunk_size):
            chunk_end = min(chunk_start + prefill_chunk_size, prompt_len)
            raw = embedding.index_select(0, input_ids[:, chunk_start:chunk_end].reshape(-1)).view(
                batch, chunk_end - chunk_start, cfg.hidden_size
            )
            if trace is not None:
                trace.record(
                    name="embedding",
                    tensor=raw,
                    token_start=chunk_start,
                    token_end=chunk_end,
                )
            normalized = rms_norm(raw, weights["input_layernorm.weight"][0], cfg.rms_norm_eps)
            for layer_idx in range(cfg.num_hidden_layers):
                kv_cache.step_prefill_chunk(layer_idx, chunk_start, chunk_end)
                normalized, raw = _transformer_prefill_layer_fused_norms(
                    normalized=normalized,
                    raw_residual=raw,
                    weights=weights,
                    cfg=cfg,
                    flash_attention=flash_attention,
                    cos=cos,
                    sin=sin,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    position=chunk_start,
                    trace=trace,
                )
            final_hidden = normalized
        if final_hidden is None:
            raise AssertionError("non-empty prompt did not produce a prefill chunk")
        if trace is not None:
            trace.record(
                name="final_norm",
                tensor=final_hidden,
                token_start=0,
                token_end=prompt_len,
            )
        kv_cache.set_cache_seqlens(
            list(resolved_row_lengths) if resolved_row_lengths is not None
            else [prompt_len] * batch)
        # Ragged: sample from each row's last REAL position; uniform: shared -1.
        # Single-chunk is guaranteed above for the ragged case, so final_hidden
        # spans the full prompt width and the gather is valid for every row.
        if resolved_row_lengths is not None:
            last_hidden = torch.stack(
                [final_hidden[r, int(resolved_row_lengths[r]) - 1] for r in range(batch)],
                dim=0,
            )
        else:
            last_hidden = final_hidden[:, -1]
        logits = (last_hidden @ embedding.T) * cfg.logit_scale
        return NmcReferencePrefillState(
            input_ids=input_ids,
            hidden=final_hidden,
            next_token=_sample_next_token(logits, temperature, top_p),
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            prompt_len=prompt_len,
            max_seq=max_seq,
        )
    except BaseException:
        kv_cache.close()
        raise


@torch.inference_mode()
def nmc_prefill_into_slot(
    *,
    session_kv: NmcPagedKvCache,
    row: int,
    input_ids: torch.Tensor,
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefill_chunk_size: int,
    temperature: float,
    top_p: float,
) -> tuple[int, int, int]:
    """Prefill a request into one paused continuous-batch KV row.

    Row-masked metadata updates and a row-sliced page table preserve every
    live row. Full cached prefix blocks are attached before suffix allocation,
    then the complete cacheable prefix is published after prefill.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.view(1, -1)
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must have shape [1, prompt_len]")
    prompt_len = int(input_ids.shape[1])
    if prompt_len < 1 or prefill_chunk_size < 1:
        raise ValueError("prompt_len and prefill_chunk_size must be positive")
    if row < 0 or row >= session_kv.batch_size:
        raise ValueError(f"row {row} is outside session batch {session_kv.batch_size}")
    _validate_sampling_params(temperature, top_p)
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must use int32 or int64 token IDs")
    embedding = weights["model.embed_tokens.weight"]
    if input_ids.device != embedding.device:
        raise ValueError("input_ids and model weights must be on the same device")

    active = [1 if batch_row == row else 0 for batch_row in range(session_kv.batch_size)]
    lengths = [prompt_len if batch_row == row else 0 for batch_row in range(session_kv.batch_size)]
    token_ids = [int(token_id) for token_id in input_ids[0].detach().cpu().tolist()]
    vocab_size = int(embedding.shape[0])
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError("prompt contains a token outside the model vocabulary")
    prefix_keys = nmc_prefix_block_keys(token_ids, session_kv.page_block)
    cached_prefix_len = _attach_prefix_row(
        session_kv,
        prefix_keys,
        row=row,
        page_block=session_kv.page_block,
        sw_size=cfg.sliding_window,
    )
    session_kv.prepare_prefill_lengths_masked(lengths, active)

    flash_attention = _require_flash_attention()
    _require_grouped_mm()
    kv_view = _NmcRowKvView(session_kv, row)
    final_hidden: torch.Tensor | None = None
    for chunk_start in range(cached_prefix_len, prompt_len, prefill_chunk_size):
        chunk_end = min(chunk_start + prefill_chunk_size, prompt_len)
        raw = embedding.index_select(
            0, input_ids[:, chunk_start:chunk_end].reshape(-1)
        ).view(1, chunk_end - chunk_start, cfg.hidden_size)
        normalized = rms_norm(raw, weights["input_layernorm.weight"][0], cfg.rms_norm_eps)
        for layer_idx in range(cfg.num_hidden_layers):
            session_kv.step_prefill_chunk_masked(
                layer_idx, chunk_start, chunk_end, active
            )
            normalized, raw = _transformer_prefill_layer_fused_norms(
                normalized=normalized,
                raw_residual=raw,
                weights=weights,
                cfg=cfg,
                flash_attention=flash_attention,
                cos=cos,
                sin=sin,
                kv_cache=kv_view,
                layer_idx=layer_idx,
                position=chunk_start,
                trace=None,
            )
        final_hidden = normalized
    if final_hidden is None:
        raise RuntimeError("prefix attachment consumed the complete prompt")

    _publish_prefix_row(session_kv, prefix_keys, row=row)
    logits = (final_hidden[:, -1] @ embedding.T) * cfg.logit_scale
    return int(_sample_next_token(logits, temperature, top_p).item()), prompt_len, cached_prefix_len


@torch.inference_mode()
def nmc_prefill_into_slots(
    *,
    session_kv: NmcPagedKvCache,
    rows: Sequence[int],
    prompt_ids: Sequence[Sequence[int]],
    weights: Mapping[str, Any],
    cfg: NmcConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    prefill_chunk_size: int,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
) -> NmcBatchedPrefillResult:
    """Pack several short suffixes into one model-weight pass without padding.

    This intentionally accepts only prompts that fit one chunk and end before
    the sliding-window boundary. Long/chunked prompts remain on the established
    serial path; silently handling them here would omit per-layer SWA eviction.
    Selected rows are cleared if any part of the packed transaction fails.
    """

    batch = len(rows)
    if batch < 2:
        raise ValueError("packed prefill requires at least two requests")
    if len(prompt_ids) != batch:
        raise ValueError("rows and prompt_ids must have matching lengths")
    if len(temperatures) != batch or len(top_ps) != batch:
        raise ValueError("sampling parameters must match the packed batch")
    resolved_rows = tuple(int(row) for row in rows)
    if len(set(resolved_rows)) != batch:
        raise ValueError("packed prefill rows must be distinct")
    if any(row < 0 or row >= session_kv.batch_size for row in resolved_rows):
        raise ValueError("packed prefill row is outside the session batch")
    if prefill_chunk_size < 1 or cfg.sliding_window <= 1:
        raise ValueError("packed prefill requires positive chunk/window limits")
    safe_prompt_limit = min(prefill_chunk_size, cfg.sliding_window - 1)
    resolved_prompts = tuple(
        tuple(int(token_id) for token_id in ids)
        for ids in prompt_ids
    )
    prompt_lengths = tuple(len(ids) for ids in resolved_prompts)
    if any(length < 1 or length > safe_prompt_limit for length in prompt_lengths):
        raise ValueError(
            "packed prefill prompts must be non-empty and fit one safe chunk "
            f"(limit={safe_prompt_limit})"
        )
    if max(prompt_lengths) > int(cos.shape[0]) or max(prompt_lengths) > int(
        sin.shape[0]
    ):
        raise ValueError("RoPE tables do not cover the complete packed prompt")
    for temperature, top_p in zip(temperatures, top_ps):
        _validate_sampling_params(float(temperature), float(top_p))
    embedding = weights["model.embed_tokens.weight"]
    vocab_size = int(embedding.shape[0])
    if any(
        token_id < 0 or token_id >= vocab_size
        for ids in resolved_prompts
        for token_id in ids
    ):
        raise ValueError("packed prompt contains a token outside the vocabulary")

    prefix_keys_by_row = tuple(
        nmc_prefix_block_keys(ids, session_kv.page_block)
        for ids in resolved_prompts
    )
    try:
        cached_prefix_lengths = _attach_prefix_rows(
            kv_cache=session_kv,
            keys_by_row=prefix_keys_by_row,
            rows=resolved_rows,
            page_block=session_kv.page_block,
            sw_size=cfg.sliding_window,
        )
        active = [
            1 if row in resolved_rows else 0
            for row in range(session_kv.batch_size)
        ]
        lengths = [0] * session_kv.batch_size
        for row, prompt_length in zip(resolved_rows, prompt_lengths):
            lengths[row] = prompt_length
        # Since every prompt is shorter than the SWA boundary, this allocates
        # every selected row/layer once. No per-layer masked step is necessary.
        session_kv.prepare_prefill_lengths_masked(lengths, active)

        suffixes = tuple(
            ids[cached_length:]
            for ids, cached_length in zip(
                resolved_prompts, cached_prefix_lengths
            )
        )
        suffix_lengths = tuple(len(ids) for ids in suffixes)
        if any(length < 1 or length > prefill_chunk_size for length in suffix_lengths):
            raise RuntimeError("prefix attachment produced an invalid packed suffix")
        cumulative = [0]
        packed_ids: list[int] = []
        packed_positions: list[int] = []
        for ids, cached_length in zip(suffixes, cached_prefix_lengths):
            packed_ids.extend(ids)
            packed_positions.extend(
                range(cached_length, cached_length + len(ids))
            )
            cumulative.append(cumulative[-1] + len(ids))

        device = embedding.device
        packed_id_tensor = torch.tensor(
            packed_ids,
            device=device,
            dtype=torch.long,
        )
        position_tensor = torch.tensor(
            packed_positions,
            device=device,
            dtype=torch.long,
        )
        cu_seqlens = torch.tensor(
            cumulative,
            device=device,
            dtype=torch.int32,
        )
        cache_seqlens = torch.tensor(
            cached_prefix_lengths,
            device=device,
            dtype=torch.int32,
        )
        packed_cos = cos.index_select(0, position_tensor)
        packed_sin = sin.index_select(0, position_tensor)
        total_tokens = len(packed_ids)
        raw = embedding.index_select(0, packed_id_tensor).view(
            1, total_tokens, cfg.hidden_size
        )
        normalized = rms_norm(
            raw,
            weights["input_layernorm.weight"][0],
            cfg.rms_norm_eps,
        )
        flash_attention = _require_flash_attention()
        _require_grouped_mm()
        kv_view = _NmcRowsKvView(session_kv, resolved_rows)
        max_seqlen = max(suffix_lengths)
        for layer_idx in range(cfg.num_hidden_layers):
            normalized, raw = _transformer_prefill_layer_packed(
                normalized=normalized,
                raw_residual=raw,
                weights=weights,
                cfg=cfg,
                flash_attention=flash_attention,
                packed_cos=packed_cos,
                packed_sin=packed_sin,
                kv_cache=kv_view,
                layer_idx=layer_idx,
                cache_seqlens=cache_seqlens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        last_indices = cu_seqlens[1:].to(torch.long) - 1
        last_hidden = normalized.view(
            total_tokens, cfg.hidden_size
        ).index_select(0, last_indices)
        logits = (last_hidden @ embedding.T) * cfg.logit_scale
        next_tokens = _sample_next_tokens(
            logits,
            temperatures=temperatures,
            top_ps=top_ps,
        )
        for row, keys in zip(resolved_rows, prefix_keys_by_row):
            _publish_prefix_row(session_kv, keys, row=row)
        return NmcBatchedPrefillResult(
            next_tokens=next_tokens,
            prompt_lengths=prompt_lengths,
            cached_prefix_lengths=cached_prefix_lengths,
            computed_tokens=total_tokens,
        )
    except BaseException:
        # Failure is rare and off the hot path. Individual frees deliberately
        # favor a simple, auditable rollback over another native batch ABI.
        for row in resolved_rows:
            session_kv.free_row(row)
        raise


def encode_prompt_token_ids(
    *, tokenizer: Any, prompt: str, add_special_tokens: bool
) -> list[int]:
    """CPU-only encode of a raw text prompt. No CUDA round-trip.

    Chat-templated strings must pass ``add_special_tokens=False``: the template
    already owns BOS (and must not grow a trailing EOS). Completions / raw
    prompts pass ``add_special_tokens=True`` so the tokenizer adds BOS.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    encoded = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
    )
    ids = encoded["input_ids"]
    # HuggingFace may return a list, tuple, or numpy array depending on version
    # and return_tensors. Normalize to a flat Python list of ints.
    if ids is None:
        raise ValueError("tokenizer produced an invalid empty prompt")
    if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
        ids = ids.tolist()
    else:
        ids = list(ids)
    if not ids:
        raise ValueError("tokenizer produced an invalid empty prompt")
    if isinstance(ids[0], (list, tuple)):
        raise ValueError("tokenizer produced a batched encoding; expected one prompt")
    return [int(token_id) for token_id in ids]


@torch.inference_mode()
def encode_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    raw_prompt: bool,
    add_special_tokens: bool,
) -> torch.Tensor:
    """Encode one text prompt with either raw tokens or the NMC chat template."""
    if raw_prompt:
        ids = encode_prompt_token_ids(
            tokenizer=tokenizer,
            prompt=prompt,
            add_special_tokens=add_special_tokens,
        )
        input_ids = torch.tensor(ids, dtype=torch.long)
    else:
        if add_special_tokens:
            raise ValueError(
                "add_special_tokens must be False when raw_prompt is False; "
                "the chat template owns special tokens"
            )
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(input_ids, str):
            input_ids = tokenizer(
                input_ids, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
        elif hasattr(input_ids, "get") and input_ids.get("input_ids") is not None:
            input_ids = input_ids["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError("tokenizer produced an invalid empty prompt")
    return input_ids.to(device=device, dtype=torch.long)


def run_reference_prefill(
    *,
    prompt: str,
    batch: int,
    weights: Mapping[str, Any],
    tokenizer: Any,
    cfg: NmcConfig,
    device: torch.device,
    dtype: torch.dtype,
    raw_prompt: bool,
    page_block: int,
    prefill_chunk_size: int,
    frac_vram_utilization: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompts: Sequence[str] | None = None,
) -> NmcReferencePrefillState:
    """Encode and prefill real NMC prompt(s) into the release paged KV cache.

    Two input modes are supported:
    - Uniform (``prompts is None``): tokenize the single ``prompt`` and
      broadcast it to ``batch`` rows (every row shares the same length).
    - Ragged (``prompts`` given): one prompt string per row, tokenized
      independently to differing lengths, right-padded to a rectangular batch
      and prefilled in a single chunk. Each row's true length becomes its
      ``cache_seqlens`` so decode (and the ``attn_drain`` per-row split policy)
      sees a genuinely ragged real batch.
    """
    if batch < 1 or max_new_tokens < 1:
        raise ValueError("batch and max_new_tokens must be positive")

    if prompts is not None:
        # ── Ragged multi-prompt real prefill ────────────────────────────────
        prompts_list = list(prompts)
        if len(prompts_list) != batch:
            raise ValueError(
                f"run_reference_prefill: got {len(prompts_list)} prompts but "
                f"batch={batch}; pass exactly one prompt per row")
        row_ids = [
            encode_prompt(
                tokenizer=tokenizer,
                prompt=p,
                device=device,
                raw_prompt=raw_prompt,
                add_special_tokens=bool(raw_prompt),
            )[0]
            for p in prompts_list
        ]
        row_lengths = [int(t.shape[0]) for t in row_ids]
        if min(row_lengths) < 1:
            raise ValueError("every ragged prompt must tokenize to >= 1 token")
        max_len = max(row_lengths)
        # Right-pad shorter rows. Pad id is arbitrary: causal attention keeps
        # padding (which sits AFTER each row's real tokens) from touching the
        # real KV, and decode never reads past the per-row cache_seqlen.
        pad_id = int(cfg.eos_token_id)
        input_ids = torch.full(
            (batch, max_len), pad_id, device=device, dtype=torch.long)
        for row, ids in enumerate(row_ids):
            input_ids[row, :row_lengths[row]] = ids
        return _nmc_prefill_forward_from_ids(
            input_ids=input_ids,
            weights=weights,
            cfg=cfg,
            device=device,
            dtype=dtype,
            page_block=page_block,
            # Force single-chunk so the full per-row residual stream survives
            # for the per-row last-real-position gather inside the forward.
            prefill_chunk_size=max_len,
            frac_vram_utilization=frac_vram_utilization,
            max_seq=max_len + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            trace=None,
            row_lengths=row_lengths,
        )

    # ── Uniform single-prompt real prefill (broadcast to batch) ─────────────
    input_ids = encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        raw_prompt=raw_prompt,
        add_special_tokens=bool(raw_prompt),
    )
    if int(input_ids.shape[0]) != 1:
        raise ValueError(
            "release prefill expects one prompt string and broadcasts it "
            f"across batch rows; got tokenized batch {int(input_ids.shape[0])}")
    if batch > 1:
        # Broadcast the prompt tokens, not the KV cache. The paged KV manager
        # still allocates independent page-table rows and physical KV pages for
        # every sequence in the batch.
        input_ids = input_ids.repeat(batch, 1).contiguous()
    return _nmc_prefill_forward_from_ids(
        input_ids=input_ids,
        weights=weights,
        cfg=cfg,
        device=device,
        dtype=dtype,
        page_block=page_block,
        prefill_chunk_size=prefill_chunk_size,
        frac_vram_utilization=frac_vram_utilization,
        max_seq=int(input_ids.shape[1]) + max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        trace=None,
    )


def run_fast_prefill(
    *,
    cfg: NmcConfig,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    fake_prompt_len: int,
    max_new_tokens: int,
    page_block: int,
    frac_vram_utilization: float,
    prefill_chunk_size: int,
    upgate_chunk: int,
    moe_upgate_chunk: int,
    warmup_iterations: int,
    timed_iterations: int,
) -> dict[str, Any]:
    """Benchmark real release prefill compute using synthetic finite weights.

    Fake prompts are drawn independently per batch row (uniform over the vocab)
    so the paged KV cache is uncorrelated across rows -- matching a multi-request
    batch rather than a broadcast of one all-zero prompt.

    The returned state owns the final iteration's KV handle.  Call
    ``result["prefill"].close()`` after consuming the result.
    """
    if batch < 1 or fake_prompt_len < 1 or max_new_tokens < 1:
        raise ValueError("batch, fake_prompt_len, and max_new_tokens must be positive")
    if warmup_iterations < 0 or timed_iterations < 1:
        raise ValueError("warmup_iterations must be non-negative and timed_iterations positive")
    weights = make_fast_nmc_weights(
        cfg=cfg,
        device=device,
        dtype=dtype,
        upgate_chunk=upgate_chunk,
        moe_upgate_chunk=moe_upgate_chunk,
    )
    # Per-row independent fake prompts.  All-zero ids made every batch row write
    # identical KV (pages were distinct, values were bitwise copies), which
    # understates attention traffic / overstates cache locality vs real
    # multi-request batches.  Uniform draws over the vocab keep rows
    # uncorrelated for --fast decode benchmarks.
    input_ids = torch.randint(
        0,
        int(cfg.vocab_size),
        (batch, fake_prompt_len),
        device=device,
        dtype=torch.long,
    )

    def one_prefill() -> NmcReferencePrefillState:
        return _nmc_prefill_forward_from_ids(
            input_ids=input_ids,
            weights=weights,
            cfg=cfg,
            device=device,
            dtype=dtype,
            page_block=page_block,
            prefill_chunk_size=prefill_chunk_size,
            frac_vram_utilization=frac_vram_utilization,
            # Keep one logical-token slot beyond the requested decode range so
            # benchmark fixtures never sit exactly on the addressable boundary.
            max_seq=fake_prompt_len + max_new_tokens + 1,
            temperature=0.0,
            top_p=1.0,
            trace=None,
        )

    for _ in range(warmup_iterations):
        warmup = one_prefill()
        torch.cuda.synchronize(device)
        warmup.close()
    measurements: list[float] = []
    prefill: NmcReferencePrefillState | None = None
    for _ in range(timed_iterations):
        if prefill is not None:
            prefill.close()
        started = time.perf_counter()
        prefill = one_prefill()
        torch.cuda.synchronize(device)
        measurements.append((time.perf_counter() - started) * 1000.0)
    if prefill is None:
        raise AssertionError("timed prefill loop did not create state")
    measurements.sort()
    prefill_ms = measurements[len(measurements) // 2]
    return {
        "weights": weights,
        "prefill": prefill,
        "prefill_ms": prefill_ms,
        "prefill_tok_s": (batch * fake_prompt_len) / max(prefill_ms / 1000.0, float.fromhex("0x1.0p-1022")),
    }


def _apply_fake_prefill_lengths(
    *, kv_cache: Any, prompt_lengths: Sequence[int],
) -> None:
    """Establish the page mapping and ``cache_seqlens`` for a synthetic prefill.

    ``prepare_prefill_lengths`` already SWA-maps pages (window-only for sliding
    layers, full history for dense). Do NOT follow with ``step_prefill_chunk``
    from 0..len -- that re-allocates the full history on sliding layers and
    undoes eviction, which is what blew the arena floor on long ragged batches.

    KV *contents* are irrelevant here (callers overwrite k_pool/v_pool with
    noise); this only establishes the page mapping and cache_seqlens.
    """
    lengths = [int(v) for v in prompt_lengths]
    kv_cache.prepare_prefill_lengths(lengths)
    kv_cache.set_cache_seqlens(lengths)


def make_synthetic_kv_prefill(
    *,
    cfg: NmcConfig,
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    fake_prompt_len: int | Sequence[int],
    max_new_tokens: int,
    page_block: int,
    frac_vram_utilization: float,
    row_active: Sequence[int] | None,
) -> NmcReferencePrefillState:
    """Build a noise-filled paged KV at ``fake_prompt_len`` without a prefill forward.

    Used by ``runner --fast --real-weight``: real checkpoint weights plus a
    synthetic KV cache so decode can measure weight/MoE traffic without running
    the prefill compute path. Pages are allocated and lengths set as if a
    prompt of ``fake_prompt_len`` had been written; K/V values are small finite
    noise (not zeros) so attention stays well-behaved.

    ``fake_prompt_len`` may be a per-row sequence, producing a RAGGED cache. The
    returned ``prompt_len`` is then ``max(lengths)``, while ``cache_seqlens``
    holds the true per-row lengths -- callers that step decode positions MUST
    read the per-row vector, not ``prompt_len``.

    ``row_active`` (optional, length ``batch``) marks continuous-batch empty
    slots: ``0`` means masked (inactive). Masked rows must have
    ``fake_prompt_len[i] == 0``; their ``kv_cache.row_active`` is written so the
    decode loop / kernel skip them. ``None`` means every row is active.

    Arena floor: price the hybrid physical demand
    ``sum_layers sum_rows pages_kept(seqlen_i)`` (full layers keep the full
    span; sliding layers keep at most ``sw_size``), not
    ``num_layers * batch * pages(max_seq)``. Convert that block count into the
    allocator's per-row ``required_max_seq`` floor. Page-table addressability
    stays at ``max_seq``.
    """
    if batch < 1 or max_new_tokens < 1:
        raise ValueError("batch and max_new_tokens must be positive")
    if isinstance(fake_prompt_len, int):
        prompt_lengths = [max(0, int(fake_prompt_len))] * batch
    else:
        prompt_lengths = [max(0, int(v)) for v in fake_prompt_len]
        if len(prompt_lengths) != batch:
            raise ValueError(
                f"fake_prompt_len has {len(prompt_lengths)} rows, expected batch={batch}")
    if row_active is None:
        active = [1] * batch
    else:
        active = [1 if int(a) != 0 else 0 for a in row_active]
        if len(active) != batch:
            raise ValueError(
                f"row_active has {len(active)} rows, expected batch={batch}")
        for i, (a, plen) in enumerate(zip(active, prompt_lengths)):
            if a == 0 and plen != 0:
                raise ValueError(
                    f"masked row {i} must have fake_prompt_len 0, got {plen}")
        if not any(active):
            raise ValueError("row_active cannot mask every row")
    max_prompt_len = max(prompt_lengths)
    max_seq = max_prompt_len + int(max_new_tokens) + 1
    # Masked rows still reserve decode headroom so a later rebind/activation
    # could use the slot; pages are not allocated for len=0 by prepare_prefill.
    row_capacities = [int(p) + int(max_new_tokens) + 1 for p in prompt_lengths]
    required_max_seq = hybrid_floor_required_max_seq(
        cfg=cfg,
        batch=int(batch),
        max_seq=int(max_seq),
        page_block=int(page_block),
        row_capacities=row_capacities,
        prefill_chunk_size=0,
    )
    cos, sin = build_rope(max_seq, cfg, device)
    kv_cache = allocate_paged_kv(
        cfg=cfg,
        batch=batch,
        max_seq=max_seq,
        page_block=page_block,
        device=device,
        dtype=dtype,
        frac_vram_utilization=frac_vram_utilization,
        required_max_seq=required_max_seq,
    )
    _apply_fake_prefill_lengths(
        kv_cache=kv_cache,
        prompt_lengths=prompt_lengths,
    )
    # Device row_active starts as all-ones from allocate; publish the mask so
    # the first decode launch (after step_decode_positions) and any code that
    # snapshots kv.row_active before the first step see inactive slots.
    # step_decode_positions will also sync the C++ host mirror from the same
    # mask on every step.
    kv_cache.row_active.copy_(
        torch.tensor(active, dtype=torch.int32, device=device),
        non_blocking=False,
    )
    kv_cache.k_pool.normal_(0.0, SYNTHETIC_WEIGHT_STDDEV)
    kv_cache.v_pool.normal_(0.0, SYNTHETIC_WEIGHT_STDDEV)
    input_ids = torch.randint(
        0,
        int(cfg.vocab_size),
        (batch, max(1, max_prompt_len)),
        device=device,
        dtype=torch.long,
    )
    next_token = torch.randint(
        0, int(cfg.vocab_size), (batch,), device=device, dtype=torch.long,
    )
    return NmcReferencePrefillState(
        input_ids=input_ids,
        hidden=torch.zeros((batch, 1, cfg.hidden_size), device=device, dtype=dtype),
        next_token=next_token,
        cos=cos,
        sin=sin,
        kv_cache=kv_cache,
        prompt_len=max_prompt_len,
        max_seq=max_seq,
    )
