"""Paged-KV cache objects over the ``kv/cache.h`` ABI.

This module owns Python-side KV state -- the shared physical K/V arena and the
per-geometry :class:`NmcPagedKvCache`.

"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

import native

# ABI and allocation constants.  Do not duplicate these literals at call sites.
KV_COMPONENT_COUNT = 2  # One K page and one V page per physical block.
MINIMUM_PHYSICAL_BLOCK_COUNT = 1
MINIMUM_LOGICAL_PAGE_COUNT = 1
BYTES_PER_GIB = 1024**3
PREFIX_CACHE_KEY_BYTES = 32  # kv/cache.h requires SHA-256 material.


@dataclass(frozen=True)
class PrefixCacheAttachment:
    """One row/logical-page attachment for the batched prefix ABI."""

    key: bytes
    row: int
    logical_block: int
    required_from_block: int


def _stream_pointer(stream: torch.cuda.Stream | None) -> int:
    """Return a CUDA stream handle; ``None`` selects the CURRENT CUDA stream.

    A ``None`` stream means "use whatever stream Torch is currently running
    on", not the legacy default stream 0.
    Defaulting to stream 0 would run KV metadata updates on a different stream
    than the surrounding Torch prefill/decode work and silently break ordering.
    ``NmcPagedKvCache`` always passes its pool's device-accurate current
    stream explicitly; this default only covers direct callers of
    :func:`create_kv_handle` that pass ``None``.
    """

    if stream is None:
        return int(torch.cuda.current_stream().cuda_stream)
    try:
        return int(stream.cuda_stream)
    except AttributeError as exc:
        raise TypeError("stream must be torch.cuda.Stream or None") from exc


def _same_cuda_device(actual: torch.device, requested: torch.device) -> bool:
    """Treat ``cuda`` as the current concrete CUDA ordinal."""
    if actual.type != requested.type:
        return False
    if requested.type != "cuda" or requested.index is not None:
        return actual.index == requested.index
    return actual.index == torch.cuda.current_device()


def _require_cuda_int32_tensor(
    tensor: torch.Tensor,
    name: str,
    dimensions: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions, got {tensor.ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_kv_pool(
    pool: torch.Tensor,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    page_block: int,
    n_kv_heads: int,
    head_dim: int,
) -> None:
    if not isinstance(pool, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_tail = (int(page_block), int(n_kv_heads), int(head_dim))
    if not pool.is_cuda or not _same_cuda_device(pool.device, device):
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if pool.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {pool.dtype}")
    if pool.ndim != 4 or tuple(pool.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape [blocks, {page_block}, {n_kv_heads}, {head_dim}]"
        )
    if pool.shape[0] < MINIMUM_PHYSICAL_BLOCK_COUNT:
        raise ValueError(f"{name} must contain at least one physical block")
    if not pool.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_handle_tensors(
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    row_active: torch.Tensor,
) -> None:
    """Check what the ``KvHandle`` constructor cannot see.

    It receives these tensors as bare device addresses, so their dtype, shape,
    device, and contiguity are checkable only here.  It does range-check the
    scalars derived from them, so this function does not repeat that.
    """

    _require_cuda_int32_tensor(page_table, "page_table", 3)
    _require_cuda_int32_tensor(cache_seqlens, "cache_seqlens", 1)
    _require_cuda_int32_tensor(row_active, "row_active", 1)
    if page_table.device != cache_seqlens.device or page_table.device != row_active.device:
        raise ValueError("page_table, cache_seqlens, and row_active must share one CUDA device")
    batch_size = int(page_table.shape[1])
    if cache_seqlens.numel() != batch_size or row_active.numel() != batch_size:
        raise ValueError("cache_seqlens and row_active must each contain one value per batch row")


def create_kv_handle(
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    row_active: torch.Tensor,
    num_phys_pages: int,
    page_block_size: int,
    start_pos: int,
    sw_size: int,
    sw_pattern: int,
) -> Any:
    """Attach release-native metadata to caller-owned CUDA page-table tensors.

    Returns a native ``kv.KvHandle``.  Prefer :func:`allocate_nmc_paged_kv_cache`,
    which pairs the handle with the tensors it borrows; a bare handle from here
    releases its KV blocks whenever the caller drops the last reference to it.

    ``sw_pattern`` is the period derived from ``layer_types``, never
    ``prefix_dense_sliding_window_pattern``.  Native code evaluates
    ``layer % sw_pattern`` even when SWA is off (``sw_size == 0``), and treats
    layers with ``(layer % sw_pattern) == 0`` as dense.
    """

    _validate_handle_tensors(page_table, cache_seqlens, row_active)
    num_layers, batch_size, max_pages = (int(dim) for dim in page_table.shape)
    kv = native.ext().kv
    desc = kv.KvHandleDesc(
        page_table=int(page_table.data_ptr()),
        cache_seqlens=int(cache_seqlens.data_ptr()),
        row_active=int(row_active.data_ptr()),
        num_layers=num_layers,
        BS=batch_size,
        max_pages_per_seq=max_pages,
        num_phys_pages=int(num_phys_pages),
        page_block_size=int(page_block_size),
        start_pos=int(start_pos),
        sw_size=int(sw_size),
        sw_pattern=int(sw_pattern),
    )
    return kv.KvHandle(desc)


@dataclass
class NmcSharedKvArena:
    """One physical K/V pool shared by every handle in this process.

    There is no library-path field guarding cross-build sharing: ``native``
    binds the process to a single build, so a second native KV manager cannot
    exist here in the first place.
    """

    num_phys_blocks: int
    k_pool: torch.Tensor
    v_pool: torch.Tensor


_NMC_SHARED_KV_ARENA: NmcSharedKvArena | None = None
_STATE_LOCK = threading.RLock()


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _vram_sized_block_count(
    required_blocks: int,
    page_block: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    frac_vram_utilization: float,
) -> int:
    """Size the physical KV pool from currently free VRAM and a hard floor."""

    if int(required_blocks) < MINIMUM_PHYSICAL_BLOCK_COUNT:
        raise ValueError("required_blocks must be positive")
    if int(page_block) < 1 or int(n_kv_heads) < 1 or int(head_dim) < 1:
        raise ValueError("page_block, n_kv_heads, and head_dim must be positive")
    fraction = float(frac_vram_utilization)
    if not 0.0 < fraction < 1.0:
        raise ValueError("frac_vram_utilization must be in (0, 1)")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("KV arena allocation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot size a paged KV arena")
    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved_device)
    page_bytes = (
        KV_COMPONENT_COUNT
        * int(page_block)
        * int(n_kv_heads)
        * int(head_dim)
        * _dtype_nbytes(dtype)
    )
    target_bytes = int(int(free_bytes) * fraction)
    target_blocks = max(MINIMUM_PHYSICAL_BLOCK_COUNT, target_bytes // page_bytes)
    if target_blocks < int(required_blocks):
        raise RuntimeError(
            "KV arena is too small after VRAM sizing: "
            f"required_blocks={int(required_blocks)} target_blocks={target_blocks} "
            f"free_gib={int(free_bytes) / BYTES_PER_GIB:.2f} "
            f"total_gib={int(total_bytes) / BYTES_PER_GIB:.2f} "
            f"frac_vram_utilization={fraction:.3f}"
        )
    return int(target_blocks)


def get_nmc_shared_kv_arena(
    required_blocks: int,
    page_block: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    frac_vram_utilization: float,
) -> NmcSharedKvArena:
    """Return the process-wide VRAM-sized NMC K/V arena.

    The native KV manager fixes its physical-pool geometry on the first handle.
    Re-sizing from free-VRAM measurements between benchmark iterations changes
    that geometry and is rejected by the ``KvHandle`` constructor. This function therefore
    allocates once, then creates independent page-table handles over one pool.
    """
    global _NMC_SHARED_KV_ARENA
    resolved_device = torch.device(device)
    with _STATE_LOCK:
        if _NMC_SHARED_KV_ARENA is not None:
            if _NMC_SHARED_KV_ARENA.num_phys_blocks < int(required_blocks):
                raise RuntimeError(
                    "fixed NMC KV arena is too small for this request; restart with a larger "
                    "--frac-vram-utilization or reduce prompt/max-context "
                    f"(required_blocks={int(required_blocks)} "
                    f"arena_blocks={_NMC_SHARED_KV_ARENA.num_phys_blocks})"
                )
            _require_kv_pool(
                _NMC_SHARED_KV_ARENA.k_pool,
                "shared k_pool",
                resolved_device,
                dtype,
                page_block,
                n_kv_heads,
                head_dim,
            )
            _require_kv_pool(
                _NMC_SHARED_KV_ARENA.v_pool,
                "shared v_pool",
                resolved_device,
                dtype,
                page_block,
                n_kv_heads,
                head_dim,
            )
            return _NMC_SHARED_KV_ARENA
        arena_blocks = _vram_sized_block_count(
            required_blocks=int(required_blocks),
            page_block=int(page_block),
            n_kv_heads=int(n_kv_heads),
            head_dim=int(head_dim),
            dtype=dtype,
            device=resolved_device,
            frac_vram_utilization=float(frac_vram_utilization),
        )
        k_pool = torch.empty(
            (arena_blocks, int(page_block), int(n_kv_heads), int(head_dim)),
            device=resolved_device,
            dtype=dtype,
        )
        v_pool = torch.empty_like(k_pool)
        _NMC_SHARED_KV_ARENA = NmcSharedKvArena(
            num_phys_blocks=int(arena_blocks),
            k_pool=k_pool,
            v_pool=v_pool,
        )
        return _NMC_SHARED_KV_ARENA


@dataclass
class NmcPagedKvCache:
    """Python-owned NMC tensors plus one release-native KV metadata handle.

    Each method below adds something the native entry point does not: the
    correct CUDA stream, the closed-handle guard, or a name that says which
    phase it belongs to. Argument validation happens natively (see the module
    docstring).
    """

    k_pool: torch.Tensor
    v_pool: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    row_active: torch.Tensor
    handle: Any  # native kv.KvHandle; None once closed.
    num_layers: int
    batch_size: int
    max_pages: int
    page_block: int
    sw_size: int
    sw_pattern: int

    def _require_open(self) -> Any:
        if self.handle is None:
            raise ValueError("this NmcPagedKvCache has been closed")
        return self.handle

    def _stream(self) -> int:
        """Match Torch prefill work rather than silently using stream zero."""

        return int(torch.cuda.current_stream(self.k_pool.device).cuda_stream)

    def close(self) -> None:
        """Drop this cache's native handle, releasing its KV blocks.

        Deterministic only because CPython refcounts: the blocks come back when
        the LAST reference to the handle goes away.  Anything else still
        holding it -- a live traceback, a descriptor that captured it -- keeps
        them allocated past this call.
        """

        self.handle = None

    def prepare_prefill_lengths(self, prompt_lengths: Sequence[int]) -> None:
        self._require_open().prepare_prefill(prompt_lengths, self._stream())

    def prepare_prefill_lengths_masked(
        self,
        prompt_lengths: Sequence[int],
        active: Sequence[int],
    ) -> None:
        """Row-targeted prepare_prefill: touch only rows where active[b] != 0.

        Used to admit a single new sequence into one row of a KV handle whose
        other rows hold live, actively-decoding sequences (continuous batching).
        """

        self._require_open().prepare_prefill_masked(
            prompt_lengths, active, self._stream()
        )

    def step_prefill_chunk(self, layer_idx: int, pos_start: int, pos_end: int) -> None:
        self._require_open().step_prefill_chunk(
            layer_idx, pos_start, pos_end, self._stream()
        )

    def step_prefill_chunk_masked(
        self,
        layer_idx: int,
        pos_start: int,
        pos_end: int,
        active: Sequence[int],
    ) -> None:
        self._require_open().step_prefill_chunk_masked(
            layer_idx, pos_start, pos_end, active, self._stream()
        )

    def set_cache_seqlens(self, lengths: Sequence[int]) -> None:
        self._require_open().set_cache_seqlens(lengths, self._stream())

    def step_decode_positions(
        self,
        positions: Sequence[int],
        active: Sequence[int],
    ) -> None:
        self._require_open().step_decode_positions(
            positions, active, self._stream()
        )

    def free_row(self, row: int) -> None:
        self._require_open().free_row(row, self._stream())

    def rebind(
        self,
        src_rows: Sequence[int],
        new_page_table: torch.Tensor,
        new_cache_seqlens: torch.Tensor,
        new_row_active: torch.Tensor,
    ) -> Any:
        """Rebind live rows into a NEW handle at new_bs (geometry BS-switch).

        Destination row ``d`` takes old row ``src_rows[d]`` (``-1`` => empty).
        KV block ownership is transferred to the new handle; the caller MUST
        close this cache after a successful rebind. Returns the new
        ``kv.KvHandle``.
        """

        _require_cuda_int32_tensor(new_page_table, "new_page_table", 3)
        _require_cuda_int32_tensor(new_cache_seqlens, "new_cache_seqlens", 1)
        _require_cuda_int32_tensor(new_row_active, "new_row_active", 1)
        if (
            int(new_page_table.shape[0]) != self.num_layers
            or int(new_page_table.shape[2]) != self.max_pages
        ):
            raise ValueError(
                "new_page_table must preserve this cache's [num_layers, *, max_pages] geometry"
            )
        if (
            new_page_table.device != self.page_table.device
            or new_cache_seqlens.device != self.page_table.device
            or new_row_active.device != self.page_table.device
        ):
            raise ValueError("new page-table tensors must remain on the cache CUDA device")
        new_batch_size = int(new_page_table.shape[1])
        if new_cache_seqlens.numel() != new_batch_size or new_row_active.numel() != new_batch_size:
            raise ValueError("new cache tensors must contain one value per destination row")
        return self._require_open().rebind(
            src_rows,
            new_batch_size,
            int(new_page_table.data_ptr()),
            int(new_cache_seqlens.data_ptr()),
            int(new_row_active.data_ptr()),
            self._stream(),
        )

    def stats(self):
        """Return the native ``KvStats`` for this handle."""

        return self._require_open().stats()

    def allocated_blocks(self) -> int:
        return int(self.stats().allocated)

    def free_blocks(self) -> int:
        return int(self._require_open().free_blocks())

    @property
    def peak_allocated(self) -> int:
        return int(self.stats().peak_allocated)

    def prefix_can_attach(
        self,
        key: bytes,
        logical_block: int,
        required_from_block: int,
    ) -> bool:
        return bool(
            self._require_open().prefix_can_attach(
                key, logical_block, required_from_block
            )
        )

    def prefix_publish(self, key: bytes, row: int, logical_block: int) -> bool:
        return bool(self._require_open().prefix_publish(key, row, logical_block))

    def prefix_attach(
        self,
        key: bytes,
        row: int,
        logical_block: int,
        required_from_block: int,
    ) -> bool:
        return bool(
            self._require_open().prefix_attach(
                key, row, logical_block, required_from_block, self._stream()
            )
        )

    def prefix_attach_batch(
        self, attachments: Sequence[PrefixCacheAttachment]
    ) -> bool:
        """Attach validated prefix pages and perform one device page-table upload."""

        if not attachments:
            return True
        return bool(
            self._require_open().prefix_attach_batch(
                b"".join(a.key for a in attachments),
                PREFIX_CACHE_KEY_BYTES,
                [a.row for a in attachments],
                [a.logical_block for a in attachments],
                [a.required_from_block for a in attachments],
                self._stream(),
            )
        )


def allocate_nmc_paged_kv_cache(
    num_layers: int,
    batch_size: int,
    max_seq: int,
    page_block: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    sw_size: int,
    sw_pattern: int,
    frac_vram_utilization: float,
    required_max_seq: int | None,
) -> NmcPagedKvCache:
    """Allocate NMC KV tensors and attach their release-native metadata.

    ``required_max_seq`` is the arena floor, not the maximum addressable
    context.  Pass ``None`` to reserve the full ``batch_size * max_seq`` floor.
    Ragged fixtures should pass a sum-equivalent length instead (see
    ``make_synthetic_kv_prefill``): the allocator still multiplies by
    ``batch_size``, so ``required_max_seq = ceil(sum_i pages(seqlen_i) / batch)
    * page_block`` makes the floor track ``sum(seqlen)`` rather than
    ``batch * max(seqlen)``.

    Serving passes its decode-reservation floor; standalone and synthetic
    callers derive an equivalent floor from their expected page demand.
    """

    for value, name in (
        (num_layers, "num_layers"),
        (batch_size, "batch_size"),
        (max_seq, "max_seq"),
        (page_block, "page_block"),
        (n_kv_heads, "n_kv_heads"),
        (head_dim, "head_dim"),
        (sw_pattern, "sw_pattern"),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if int(sw_size) < 0:
        raise ValueError("sw_size must be non-negative")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("NMC paged KV cache requires a CUDA device")
    max_pages = max(
        MINIMUM_LOGICAL_PAGE_COUNT,
        (int(max_seq) + int(page_block) - 1) // int(page_block),
    )
    floor_seq = int(max_seq) if required_max_seq is None else int(required_max_seq)
    floor_seq = max(MINIMUM_LOGICAL_PAGE_COUNT, min(floor_seq, int(max_seq)))
    floor_pages = max(
        MINIMUM_LOGICAL_PAGE_COUNT,
        (floor_seq + int(page_block) - 1) // int(page_block),
    )
    required_blocks = int(num_layers) * int(batch_size) * floor_pages
    arena = get_nmc_shared_kv_arena(
        required_blocks=required_blocks,
        page_block=page_block,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=resolved_device,
        frac_vram_utilization=frac_vram_utilization,
    )
    page_table = torch.full(
        (int(num_layers), int(batch_size), max_pages),
        -1,
        device=resolved_device,
        dtype=torch.int32,
    )
    cache_seqlens = torch.zeros(
        (int(batch_size),),
        device=resolved_device,
        dtype=torch.int32,
    )
    row_active = torch.ones(
        (int(batch_size),),
        device=resolved_device,
        dtype=torch.int32,
    )
    _require_kv_pool(
        arena.k_pool, "k_pool", resolved_device, dtype, page_block, n_kv_heads, head_dim
    )
    _require_kv_pool(
        arena.v_pool, "v_pool", resolved_device, dtype, page_block, n_kv_heads, head_dim
    )
    handle = create_kv_handle(
        page_table,
        cache_seqlens,
        row_active,
        arena.num_phys_blocks,
        page_block,
        0,
        sw_size,
        sw_pattern,
    )
    return NmcPagedKvCache(
        k_pool=arena.k_pool,
        v_pool=arena.v_pool,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        row_active=row_active,
        handle=handle,
        num_layers=int(num_layers),
        batch_size=int(batch_size),
        max_pages=max_pages,
        page_block=int(page_block),
        sw_size=int(sw_size),
        sw_pattern=int(sw_pattern),
    )
