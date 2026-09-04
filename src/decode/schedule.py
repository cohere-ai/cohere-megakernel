"""Persistent NMC decode schedule construction and execution.

Public entry points borrow prefill, weights, KV state, and their CUDA tensors
for the duration of each call. Fast decoding is greedy-only; synthetic weights
are performance fixtures, not a model-correctness path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import native
import torch

if TYPE_CHECKING:
    # The dynamically loaded nanobind module has no importable type stub.
    NmcLaunchDesc = Any

logger = logging.getLogger(__name__)


# NMC model ABI constants; keep synchronized with decode/megakernel.cuh.
NMC_INSTRUCTION_WORDS = 32
# Packed MoE task record width; route_finalize emits one int4 per task.
MOE_TASK_RECORD_INTS = 4
NMC_NUM_EXPERTS = 128
NMC_TOPK = 8
NMC_EXPERT_DFF = 768
NMC_HIDDEN_SIZE = 2_048
NMC_HEAD_DIM = 128
NMC_NUM_ATTENTION_HEADS = 32
NMC_NUM_KV_HEADS = 4
NMC_RMS_NORM_EPS = 1.0e-6
NMC_VOCAB_SIZE = 262_144  # decode/launch.cuh hard-codes the LM-head GL width.
DEFAULT_MAX_ATTN_SPLITS = 32
DEFAULT_MIN_ATTN_CHUNK = 512
# Precomputed context buckets avoid rebuilding schedules during decode.
NMC_CONTEXT_BUCKET_MIN = 1024
SUPPORTED_BATCH_SIZES = (1, 2, 4, 8)
ATTN_TILE_BATCH_SHIFT = 16  # Packed tile-coordinate ABI in decode/megakernel.cuh.
ATTN_SPLIT_CAP_SHORT_CONTEXT = 16  # RR policy's <= 65,537-token cap.
ATTN_SPLIT_SHORT_CONTEXT_TOKENS = 65_537  # Inclusive current-token attention span.
# Makes drain/combine read live queue lengths and per-row split counts.
# Must match decode/megakernel.cuh::ATTN_DYNAMIC_SENTINEL.
ATTN_DYNAMIC_SENTINEL = 0
# The unconditional combine wave requires two arrivals. With one split,
# attn_decode bypasses bar_combine and the combine wait hangs.
ATTN_DRAIN_MIN_SPLITS = 2
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _div_ceil(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _moe_row_blocks_per_layer(batch: int, moe_bm: int) -> int:
    """Number of padded expert row blocks per MoE layer.

    Every expert reserves ``ceil(batch/moe_bm)`` row blocks so both
    fine-grained MoE barrier arrays have one slot per (expert, row-block).
    """
    rows_per_expert = _div_ceil(int(batch), int(moe_bm)) * int(moe_bm)
    return NMC_NUM_EXPERTS * (rows_per_expert // int(moe_bm))


def _max_moe_row_tiles_total(batch: int, moe_bm: int) -> int:
    """Upper bound on dropless top-k expert row tiles for one decode layer."""
    total_routes = int(batch) * NMC_TOPK
    active_experts = min(NMC_NUM_EXPERTS, total_routes)
    routes_after_first_tiles = max(0, total_routes - active_experts)
    extra_rows_per_expert = max(0, _div_ceil(batch, moe_bm) - 1)
    return active_experts + min(
        active_experts * extra_rows_per_expert, routes_after_first_tiles // moe_bm)


def _ptr(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else int(tensor.data_ptr())


# NmcLaunchDesc fields whose value is the device pointer of the like-named entry
# in the decode state's tensor dict -- two thirds of the descriptor.
#
# `k_cache` / `v_cache` are excluded: they alias the `K_pool` / `V_pool` tensors
# under different field names, and are assigned explicitly to keep the aliasing
# visible.
_LAUNCH_DESC_TENSOR_FIELDS: tuple[str, ...] = (
    "W_upgate", "W_down", "W_qkv", "W_oproj", "W_lmhead", "W_router",
    "W_moe_upgate", "W_moe_down", "K_pool", "V_pool",
    "page_table", "cache_seqlens", "row_active",
    "cos_table", "sin_table",
    "upgate_scratch", "silu_out", "q_out", "o_proj_in", "x_resid", "o_partial",
    "lse_partial", "lm_logits", "rmsnorm_gamma", "x_raw", "x_attn", "x_ffn",
    "router_logits", "moe_x", "moe_hidden", "moe_down_out", "topk_experts",
    "topk_scores", "topk_local_slots", "route_row_for_token",
    "routed_token_ids", "routed_scores", "expert_counts", "expert_offsets",
    "moe_up_task_words", "moe_down_task_words", "moe_up_task_count",
    "moe_down_task_count", "moe_up_task_head", "moe_down_task_head",
    "bar_upgate", "bar_silu", "bar_ffn_down", "bar_qkv", "bar_combine",
    "bar_attn", "bar_oproj", "bar_layer", "bar_router", "bar_topk",
    "bar_route", "bar_gather", "bar_moe_upgate", "bar_moe_down", "prof_buf",
    "attn_queue_words", "attn_queue_heads", "attn_num_splits",
)


def _descriptor_field_names(descriptor_type: type) -> set[str]:
    """Field names a bound descriptor class exposes."""
    return {
        name for name in dir(descriptor_type)
        if not name.startswith("_")
        and isinstance(getattr(descriptor_type, name), property)
    }


def _build_launch_desc(
    *,
    library: ModuleType,
    tensors: Mapping[str, torch.Tensor],
    kv: Any,
    cfg: Any,
    inst: torch.Tensor,
    counts: torch.Tensor,
    device: torch.device,
    num_sms: int,
    layers: int,
    batch: int,
    d: int,
    max_routed: int,
    max_tasks: int,
) -> Any:
    """Populate ``NmcLaunchDesc`` by field name.

    Positional construction would let a field inserted in decode/abi.h shift
    every subsequent value. Nanobind rejects unknown field names, so assignment
    by name turns that ABI drift into an immediate error.
    """
    desc = library.launch.NmcLaunchDesc()
    for name in _LAUNCH_DESC_TENSOR_FIELDS:
        setattr(desc, name, _ptr(tensors[name]))

    scalars: dict[str, Any] = {
        "num_sms": int(num_sms),
        "stream_u64": int(torch.cuda.current_stream(device).cuda_stream),
        "inst_buf": _ptr(inst),
        "num_inst_per_sm": _ptr(counts),
        "max_inst": int(inst.shape[1]),
        "num_phys_pages": int(kv.k_pool.shape[0]),
        "max_pages_per_seq": int(kv.max_pages),
        "page_block_size": int(kv.page_block),
        # Aliases of the pool tensors above, under the kernel's own names.
        "k_cache": _ptr(tensors["K_pool"]),
        "v_cache": _ptr(tensors["V_pool"]),
        "max_routed": int(max_routed),
        "max_moe_tasks": int(max_tasks),
        "num_layers": int(layers),
        "BS": int(batch),
        "D": int(d),
        "Dff": int(cfg.prefix_dense_intermediate_size),
        "Hq": int(cfg.num_attention_heads),
        "Hkv": int(cfg.num_key_value_heads),
        "head_dim": int(cfg.head_dim),
        "num_splits": max(1, int(num_sms)),
        "timing": 1,
        # Refreshed per step, or left 0 when the attention drain is off.
        "attn_queue_len": 0,
    }
    for name, value in scalars.items():
        setattr(desc, name, value)

    # Reject fields added to decode/abi.h without an explicit value here.
    assigned = set(_LAUNCH_DESC_TENSOR_FIELDS) | set(scalars)
    unassigned = _descriptor_field_names(type(desc)) - assigned
    if unassigned:
        raise RuntimeError(
            f"NmcLaunchDesc fields never assigned: {sorted(unassigned)}. They "
            f"were added to src/decode/abi.h; give them a value in "
            f"_build_launch_desc (zero is fine, but say so explicitly)."
        )
    return desc


def _dump_nmc_device_ptrs(
    *,
    path: str,
    tensors: Mapping[str, torch.Tensor],
    desc: NmcLaunchDesc,
    tiling: NmcTiling,
    moe_row_blocks: int,
    max_routed: int,
    max_tasks: int,
    variant: NmcDecodeVariant | None,
    inst_buf_ptr: int,
    num_inst_per_sm_ptr: int,
    max_inst: int,
) -> None:
    """Write device addresses and geometry needed to inspect cuda-gdb hangs.

    Live counters are read in-session with cuda-gdb. Barrier/task buffers are
    process-stable, while instruction buffers are recorded per schedule variant.
    """
    layers = int(desc.num_layers)
    batch = int(desc.BS)
    d = int(desc.D)
    down_bn = int(tiling.moe_down.bn)
    down_split = int(tiling.moe_down.split_k)
    down_n_tiles = (d + down_bn - 1) // down_bn
    down_arrives_per_row_block = down_split * down_n_tiles

    def line(name: str, addr: int, note: str = "") -> str:
        suffix = f"  # {note}" if note else ""
        return f"{name:28s} 0x{addr:016x}{suffix}\n"

    names_ptrs = [
        ("moe_up_task_words", "moe_up_task_words"),
        ("moe_down_task_words", "moe_down_task_words"),
        ("moe_up_task_count", "moe_up_task_count"),
        ("moe_down_task_count", "moe_down_task_count"),
        ("moe_up_task_head", "moe_up_task_head"),
        ("moe_down_task_head", "moe_down_task_head"),
        ("route_row_for_token", "route_row_for_token"),
        ("routed_token_ids", "routed_token_ids"),
        ("routed_scores", "routed_scores"),
        ("expert_counts", "expert_counts"),
        ("expert_offsets", "expert_offsets"),
        ("topk_experts", "topk_experts"),
        ("bar_upgate", "bar_upgate"),
        ("bar_silu", "bar_silu"),
        ("bar_ffn_down", "bar_ffn_down"),
        ("bar_qkv", "bar_qkv"),
        ("bar_combine", "bar_combine"),
        ("bar_attn", "bar_attn"),
        ("bar_oproj", "bar_oproj"),
        ("bar_layer", "bar_layer"),
        ("bar_router", "bar_router"),
        ("bar_topk", "bar_topk"),
        ("bar_route", "bar_route"),
        ("bar_gather", "bar_gather"),
        ("bar_moe_upgate", "bar_moe_upgate"),
        ("bar_moe_down", "bar_moe_down"),
    ]

    parts: list[str] = []
    parts.append("# NMC device pointer map for cuda-gdb\n")
    if variant is not None:
        parts.append(
            f"# schedule variant: bucket_upper={int(variant.bucket_upper)} "
            f"split_signature={variant.split_signature!r}\n"
        )
        parts.append(f"# schedule_key={variant.schedule_key!r}\n")
        parts.append(f"# tiling_key={variant.tiling_key!r}\n")
        parts.append(
            f"# schedule_chunks={[c.label for c in variant.schedule_chunks]!r} "
            f"max_inst={max_inst}\n"
        )
    parts.append(
        f"# geometry: layers={layers} BS={batch} D={d} Hkv={int(desc.Hkv)} "
        f"Hq={int(desc.Hq)} moe_row_blocks={moe_row_blocks} "
        f"max_routed={max_routed} max_moe_tasks={max_tasks}\n"
    )
    parts.append(
        f"# moe_down tiling: BM={tiling.moe_down.bm} BN={down_bn} "
        f"SPLIT_K={down_split} -> arrives/row_block={down_arrives_per_row_block}\n"
    )
    parts.append(
        f"# moe_upgate tiling: BM={tiling.moe_upgate.bm} BN={tiling.moe_upgate.bn} "
        f"SPLIT_K={tiling.moe_upgate.split_k}\n"
    )
    parts.append(
        "# gdb: layer L count/head -> x/1wu moe_down_task_count+4*L ; "
        "x/1wu moe_down_task_head+4*L\n"
    )
    parts.append(
        "# gdb: bar cell idx=(cell-bar_moe_down)/4 ; "
        "local=idx-L*moe_row_blocks\n"
    )
    parts.append(
        "# known NmcGlobals const (bs8 JIT observed): bar_layer c[0x0][0x2be8], "
        "bar_moe_down c[0x0][0x2c18], moe_down_task_count c[0x0][0x2b80], "
        "moe_down_task_head c[0x0][0x2b90]\n"
    )
    parts.append(
        "# NOTE: barrier/task ptrs are shared across schedule variants; only "
        "inst_buf / num_inst_per_sm differ per file.\n"
    )
    parts.append("\n")

    for name, tensor_key in names_ptrs:
        if tensor_key is not None and tensor_key in tensors:
            addr = _ptr(tensors[tensor_key])
            numel = int(tensors[tensor_key].numel())
            parts.append(line(name, addr, f"numel={numel}"))
        else:
            parts.append(line(name, 0, "MISSING"))

    parts.append(line("inst_buf", int(inst_buf_ptr), f"variant chunk max_inst={max_inst}"))
    parts.append(line("num_inst_per_sm", int(num_inst_per_sm_ptr), "variant chunk"))

    # Desc-side cross-check for shared buffers (inst_* may differ from active desc).
    parts.append("\n# desc field cross-check (shared buffers; inst_* = active desc)\n")
    for field in (
        "moe_down_task_count", "moe_down_task_head", "moe_down_task_words",
        "bar_moe_down", "bar_moe_upgate", "bar_ffn_down", "bar_layer",
        "bar_route", "bar_gather", "bar_router", "bar_qkv",
        "route_row_for_token", "expert_counts",
        "inst_buf", "num_inst_per_sm",
    ):
        parts.append(line(f"desc.{field}", int(getattr(desc, field))))

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(parts)
    logger.info("wrote NMC device pointer map to %s", path)


def _dump_nmc_device_ptrs_for_registry(
    *,
    tensors: Mapping[str, torch.Tensor],
    desc: NmcLaunchDesc,
    tiling: NmcTiling,
    decode_registry: NmcDecodeVariantRegistry,
    moe_row_blocks: int,
    max_routed: int,
    max_tasks: int,
) -> None:
    """Write one ptr map under ``cwd/dump/bs{BS}/`` per unique schedule variant.

    The continuous-batch session builds one geometry per supported batch size;
    isolating by ``BS`` avoids later ``make_decode_state`` calls clobbering the
    maps for the live hung kernel. Also refreshes ``cwd/dump/INDEX.txt`` listing
    every geometry directory present, and self-checks each written file.

    No-op unless --debug is set: these maps exist purely to feed cuda-gdb the
    device addresses of a live hung kernel, and writing them on every geometry
    build litters the production working directory.
    """
    if not native.debug_enabled():
        return
    batch = int(desc.BS)
    root_dump = os.path.join(os.getcwd(), "dump")
    dump_dir = os.path.join(root_dump, f"bs{batch}")
    os.makedirs(dump_dir, exist_ok=True)

    # Drop legacy flat dumps that predate dump/bs{N}/ (do not touch other files).
    for name in os.listdir(root_dump):
        if name.startswith("nmc_dev_ptrs_") and name.endswith(".txt"):
            flat = os.path.join(root_dump, name)
            if os.path.isfile(flat):
                try:
                    os.remove(flat)
                except OSError:
                    pass

    unique = decode_registry.unique_schedule_variants()
    key_to_file: dict[tuple[Any, ...], str] = {}
    bar_qkv = _ptr(tensors["bar_qkv"])
    bar_layer = _ptr(tensors["bar_layer"])
    bar_ffn = _ptr(tensors["bar_ffn_down"])
    moe_down_count = _ptr(tensors["moe_down_task_count"])
    written: list[str] = []

    for i, variant in enumerate(unique):
        chunk = variant.schedule_chunks[0]
        fname = f"nmc_dev_ptrs_v{i}_bucket{int(variant.bucket_upper)}.txt"
        path = os.path.join(dump_dir, fname)
        key_to_file[variant.schedule_key] = fname
        _dump_nmc_device_ptrs(
            path=path,
            tensors=tensors,
            desc=desc,
            tiling=tiling,
            moe_row_blocks=moe_row_blocks,
            max_routed=max_routed,
            max_tasks=max_tasks,
            variant=variant,
            inst_buf_ptr=_ptr(chunk.inst_buf),
            num_inst_per_sm_ptr=_ptr(chunk.inst_counts),
            max_inst=int(chunk.inst_buf.shape[1]),
        )
        # Self-check: file must encode this geometry's BS and live bar bases.
        with open(path, encoding="utf-8") as f:
            body = f.read()
        errors: list[str] = []
        if f"BS={batch}" not in body:
            errors.append(f"missing BS={batch}")
        for label, addr in (
            ("bar_qkv", bar_qkv),
            ("bar_layer", bar_layer),
            ("bar_ffn_down", bar_ffn),
            ("moe_down_task_count", moe_down_count),
        ):
            needle = f"0x{addr:016x}"
            if needle not in body:
                errors.append(f"missing {label}={needle}")
        inst_needle = f"0x{_ptr(chunk.inst_buf):016x}"
        if inst_needle not in body:
            errors.append(f"missing variant inst_buf={inst_needle}")
        if errors:
            raise RuntimeError(
                f"NMC ptr dump self-check failed for {path}: " + "; ".join(errors))
        written.append(path)

    index_path = os.path.join(dump_dir, "INDEX.txt")
    index_lines = [
        "# NMC schedule-variant pointer dump index\n",
        f"# dump_dir={dump_dir}\n",
        f"# BS={batch}\n",
        f"# unique_schedules={len(unique)} registry_buckets={len(decode_registry.variants)}\n",
        f"# context_range=[{decode_registry.min_context_len}, "
        f"{decode_registry.max_context_len}] attn_drain={decode_registry.attn_drain}\n",
        f"# bar_qkv=0x{bar_qkv:016x}\n",
        f"# bar_layer=0x{bar_layer:016x}\n",
        f"# bar_ffn_down=0x{bar_ffn:016x}\n",
        f"# moe_down_task_count=0x{moe_down_count:016x}\n",
        "\n# unique schedules\n",
    ]
    for i, variant in enumerate(unique):
        index_lines.append(
            f"v{i}  file={key_to_file[variant.schedule_key]}  "
            f"bucket_upper={int(variant.bucket_upper)}  "
            f"split_signature={variant.split_signature!r}\n"
        )
    index_lines.append("\n# every context bucket -> unique schedule file\n")
    for variant in decode_registry.variants:
        fname = key_to_file[variant.schedule_key]
        index_lines.append(
            f"bucket_upper<={int(variant.bucket_upper)}  ->  {fname}\n"
        )
    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(index_lines)

    # Top-level index across all geometries (rebuilt from directories present).
    top_index = os.path.join(root_dump, "INDEX.txt")
    geo_lines = [
        "# NMC device-pointer dumps by geometry\n",
        f"# root={root_dump}\n",
        "# layout: dump/bs{N}/nmc_dev_ptrs_v*_bucket*.txt + dump/bs{N}/INDEX.txt\n",
        "# pick bsN matching the hung JIT (log line: using bucket (bs=N, ...))\n",
        "\n",
    ]
    for name in sorted(os.listdir(root_dump)):
        sub = os.path.join(root_dump, name)
        if not (name.startswith("bs") and os.path.isdir(sub)):
            continue
        sub_index = os.path.join(sub, "INDEX.txt")
        n_maps = len([
            f for f in os.listdir(sub)
            if f.startswith("nmc_dev_ptrs_") and f.endswith(".txt")
        ])
        geo_lines.append(f"{name}/  maps={n_maps}  index={sub_index}\n")
        if os.path.isfile(sub_index):
            with open(sub_index, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("# BS=") or line.startswith("# bar_qkv="):
                        geo_lines.append(f"  {line}")
    with open(top_index, "w", encoding="utf-8") as f:
        f.writelines(geo_lines)

    logger.info(
        "NMC ptr dump OK geometry BS=%d -> %d maps under %s (top index %s)",
        batch, len(written), dump_dir, top_index)
    for path in written:
        logger.info("  %s", path)


def _validate_nmc_model_abi(cfg: Any) -> None:
    """Reject config shapes that the current C++ NMC ABI hard-codes."""
    exact_fields = {
        "num_experts": NMC_NUM_EXPERTS,
        "num_experts_per_tok": NMC_TOPK,
        "intermediate_size": NMC_EXPERT_DFF,
        "hidden_size": NMC_HIDDEN_SIZE,
        "head_dim": NMC_HEAD_DIM,
        "num_attention_heads": NMC_NUM_ATTENTION_HEADS,
        "num_key_value_heads": NMC_NUM_KV_HEADS,
        "vocab_size": NMC_VOCAB_SIZE,
    }
    mismatches = [
        f"{field}={getattr(cfg, field, None)!r} (expected {expected})"
        for field, expected in exact_fields.items()
        if int(getattr(cfg, field, -1)) != expected
    ]
    if float(getattr(cfg, "rms_norm_eps", float("nan"))) != NMC_RMS_NORM_EPS:
        mismatches.append(
            f"rms_norm_eps={getattr(cfg, 'rms_norm_eps', None)!r} "
            f"(expected {NMC_RMS_NORM_EPS})")
    if mismatches:
        raise ValueError(
            "mk_release NMC kernel ABI does not support this config: "
            + ", ".join(mismatches))


@dataclass(frozen=True)
class GemmTiling:
    """One JIT GEMM configuration mirrored by ``decode/runtime.cu::emit_op_cfg``.

    ``prefetch_stages`` is the number of initial pipeline stages whose weight
    TMAs are issued before the input dependency wait. Zero selects the legacy
    path. For dynamic MoE drains that legacy path still issues the first
    producer-owned weight tile before its per-task wait.
    """

    bm: int
    bn: int
    bk: int
    num_cwg: int
    stages: int
    prefetch_stages: int
    split_k: int
    m_rows: int
    direct_store: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "bm": self.bm, "bn": self.bn, "bk": self.bk,
            "num_cwg": self.num_cwg,
            "stages": self.stages, "prefetch_stages": self.prefetch_stages,
            "split_k": self.split_k,
            "m_rows": self.m_rows,
            "direct_store": self.direct_store,
        }


@dataclass(frozen=True)
class NmcTiling:
    """The release JIT configuration for one fixed decode batch size."""

    batch_size: int
    num_warpgroups: int
    num_warps: int
    num_threads: int
    inst_ring: int
    upgate: GemmTiling
    down: GemmTiling
    qkv: GemmTiling
    oproj: GemmTiling
    lmhead: GemmTiling
    router: GemmTiling
    moe_upgate: GemmTiling
    moe_down: GemmTiling
    # A/B arm: MoE down-drain scatter-reduces into x_ffn and skips MOE_COMBINE.
    moe_combine_atomic_tma: bool = False

    def as_json(self) -> str:
        payload = {
            "schema_version": 1, "batch_size": self.batch_size,
            "num_warpgroups": self.num_warpgroups,
            "num_warps": self.num_warps, "num_threads": self.num_threads,
            "inst_ring": self.inst_ring,
            **{name: getattr(self, name).as_json() for name in (
                "upgate", "down", "qkv", "oproj", "lmhead", "router",
                "moe_upgate", "moe_down",
            )},
            "moe_combine_atomic_tma": bool(self.moe_combine_atomic_tma),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def _op(
    *,
    batch: int,
    bn: int,
    num_cwg: int,
    stages: int,
    prefetch_stages: int,
    split_k: int,
    direct_store: bool,
) -> GemmTiling:
    if not 0 <= prefetch_stages <= stages:
        raise ValueError(
            f"prefetch_stages must be in [0, stages], got "
            f"prefetch_stages={prefetch_stages}, stages={stages}"
        )
    return GemmTiling(
        bm=16,
        bn=bn,
        bk=64,
        num_cwg=num_cwg,
        stages=stages,
        prefetch_stages=prefetch_stages,
        split_k=split_k,
        m_rows=batch,
        direct_store=direct_store,
    )


def default_nmc_tiling_for_bs(*, batch: int, moe_combine_atomic_tma: bool) -> NmcTiling:
    """Return the tiling table.
    """
    if int(batch) not in SUPPORTED_BATCH_SIZES:
        raise ValueError(f"unsupported NMC JIT batch size {batch}")
    batch = int(batch)
    # 512-step seq2K/4K tuning: multi-stage MoE prefetch regresses TinyM.
    # BS1 keeps the legacy path; one explicit stage wins for BS2/4/8.
    moe_prefetch_stages = 0 if batch == 1 else 1
    upgate = _op(batch=batch, bn=128, num_cwg=2, stages=10, prefetch_stages=0, split_k=1,
                 direct_store=False)
    down = _op(batch=batch, bn=64, num_cwg=1, stages=10, prefetch_stages=0, split_k=2,
               direct_store=True)
    qkv = _op(batch=batch, bn=64, num_cwg=1, stages=10, prefetch_stages=10, split_k=1,
              direct_store=False)
    oproj = _op(batch=batch, bn=64, num_cwg=1, stages=8, prefetch_stages=0, split_k=4,
                direct_store=False)
    lmhead = _op(batch=batch, bn=64, num_cwg=1, stages=10, prefetch_stages=0, split_k=1,
                 direct_store=True)

    router = _op(batch=batch, bn=64, num_cwg=1, stages=10, prefetch_stages=10, split_k=4,
                 direct_store=False)
    moe_upgate = _op(batch=batch, bn=128, num_cwg=2, stages=10,
                     prefetch_stages=moe_prefetch_stages, split_k=1,
                     direct_store=False)
    moe_down = _op(batch=batch, bn=128, num_cwg=2, stages=10,
                   prefetch_stages=moe_prefetch_stages, split_k=1,
                   direct_store=True)
    return NmcTiling(
        batch_size=batch, num_warpgroups=3,
        num_warps=12, num_threads=384, inst_ring=2, upgate=upgate, down=down,
        qkv=qkv, oproj=oproj, lmhead=lmhead, router=router,
        moe_upgate=moe_upgate, moe_down=moe_down,
        moe_combine_atomic_tma=bool(moe_combine_atomic_tma),
    )


class NmcOp:
    NOP, FFN_DOWN, QKV_PROJ, ATTN_DECODE, ATTN_COMBINE, O_PROJ, LM_HEAD = range(7)
    FFN_UPGATE_ACT, ATTN_DRAIN, ROUTER_GEMM, ROUTER_TOPK, ROUTE_FINALIZE = range(7, 12)
    MOE_GATHER, MOE_UPGATE_ACT_DRAIN, MOE_DOWN_DRAIN, MOE_COMBINE, ADD_RMSNORM = range(12, 17)
    # Ablation-only inter-wave grid barrier; not emitted by the production schedule.
    GRID_SYNC = 17


# Runtime descriptors are bound directly from decode/abi.h by bindings/launch.cpp.


def _native_error(library: ModuleType, fallback: str) -> RuntimeError:
    """Build an error for decode-service ``run()`` from its thread-local slot.

    Unlike other native entry points, ``run()`` returns a status so callers can
    distinguish ordinary failure from watchdog expiry.
    """
    return RuntimeError(library.launch.last_error() or fallback)


def _compact_profile_span_ms(*, path: str) -> float:
    """Return earliest-start to latest-end span from a compact Perfetto trace."""
    with open(path, "r", encoding="utf-8") as trace_file:
        payload = json.load(trace_file)
    raw_events = payload.get("traceEvents")
    if not isinstance(raw_events, list):
        raise TypeError("profiler trace has no traceEvents list")
    ranges = [
        event for event in raw_events
        if isinstance(event, dict)
        and event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur"), (int, float))
    ]
    if not ranges:
        raise RuntimeError("profiler trace contains no complete range events")
    start_us = min(float(event["ts"]) for event in ranges)
    end_us = max(float(event["ts"]) + float(event["dur"]) for event in ranges)
    return (end_us - start_us) / 1000.0


def jit_compile(
    *, library: ModuleType, batch: int, tiling: NmcTiling,
    enable_profiler: bool,
) -> Any:
    """Compile one persistent kernel. Returns a native ``launch.JitKernel``.

    The kernel is unloaded when the last holder drops it, which includes any
    descriptor whose ``jit_handle`` names it -- so there is no destroy to
    forget, and no window where a descriptor points at an unloaded .so.
    """
    return library.launch.JitKernel.compile(
        bs=int(batch),
        enable_profiler=bool(enable_profiler),
        repo_root=_REPO_ROOT,
        config_json=tiling.as_json(),
    )


def _zeros() -> list[int]:
    return [0] * NMC_INSTRUCTION_WORDS


def _grid_sync_inst() -> list[int]:
    """Encode a GRID_SYNC instruction (opcode only; remaining words unused)."""
    w = _zeros()
    w[0] = NmcOp.GRID_SYNC
    return w


# Fixed instruction-word ABI mirrored in decode/megakernel.cuh and launch.cuh;
# changing an offset requires changing the kernel.
class gemm_field:
    OPCODE = 0
    LAYER = 1
    TILE_M = 2
    TILE_N = 3
    SPLIT = 4
    K_TILES = 5
    WAIT_BAR_IDX = 6
    WAIT_TARGET = 7
    PRODUCE_BAR_IDX = 8
    NUM_TILES = 9
    TILE_IDS = 10


class router_topk_field:
    OPCODE = 0
    LAYER = 1
    ROW = 2
    WAIT_TARGET = 3


class route_finalize_field:
    OPCODE = 0
    LAYER = 1


class qkv_field:
    OPCODE = 0
    LAYER = 1
    TILE_M = 2
    TILE_N = 3
    GEMM_IDX = 4
    NEEDS_ROPE = 5
    K_TILES = 6
    WAIT_BAR_IDX = 7
    WAIT_TARGET = 8
    PRODUCE_BAR_IDX = 9


class attn_field:
    OPCODE = 0
    LAYER = 1
    BATCH_IDX = 2
    KV_HEAD_IDX = 3
    SPLIT_IDX = 4
    NUM_SPLITS = 5
    WINDOW_SIZE = 6
    WAIT_TARGET = 8
    NUM_TILES = 10
    TILE_IDS = 11


class attn_drain_field:
    """ATTN_DRAIN layout mirrored in decode/megakernel.cuh.

    Each instruction claims queued ATTN_DECODE work. The dynamic sentinel makes
    QUEUE_LEN come from ``g.attn_queue_len``.
    """

    OPCODE = 0
    LAYER = 1
    QUEUE_OFFSET = 2
    QUEUE_LEN = 3


class combine_field:
    """ATTN_COMBINE layout mirrored in decode/megakernel.cuh.

    The dynamic sentinel makes the reduction and wait bounds come from the
    current row's ``g.attn_num_splits`` entry.
    """

    OPCODE = 0
    LAYER = 1
    BATCH_IDX = 2
    KV_HEAD_IDX = 3
    NUM_SPLITS = 4
    WAIT_BAR_IDX = 5
    WAIT_TARGET = 6
    PRODUCE_BAR_IDX = 7


class moe_gather_field:
    OPCODE = 0
    LAYER = 1
    ROW = 2


class rmsnorm_field:
    OPCODE = 0
    LAYER = 1
    ROW = 2
    # >0: wait bar_ffn_down[layer] >= target (dense / baseline MoE combine).
    # -1: atomic-TMA MoE-combine sentinel (bar_route + fine-grained bar_moe_down).
    #  0: skip FFN wait.
    WAIT_FFN_TARGET = 3
    WAIT_OPROJ_TARGET = 4
    PRODUCE_BAR_IDX = 5


def _pack_gemm_tile_coord(tile_m: int, tile_n: int) -> int:
    return (int(tile_m) << 16) | int(tile_n)


def _pack_attn_tile(batch_idx: int, kv_head_idx: int) -> int:
    return (int(batch_idx) << ATTN_TILE_BATCH_SHIFT) | int(kv_head_idx)


def _gemm_inst(
    opcode: int,
    *,
    layer: int,
    tile_m: int,
    tile_n: int,
    split: int,
    k_tiles: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
) -> list[int]:
    w = _zeros()
    w[gemm_field.OPCODE] = int(opcode)
    w[gemm_field.LAYER] = int(layer)
    w[gemm_field.TILE_M] = int(tile_m)
    w[gemm_field.TILE_N] = int(tile_n)
    w[gemm_field.SPLIT] = int(split)
    w[gemm_field.K_TILES] = int(k_tiles)
    w[gemm_field.WAIT_BAR_IDX] = int(wait_bar_idx)
    w[gemm_field.WAIT_TARGET] = int(wait_target)
    w[gemm_field.PRODUCE_BAR_IDX] = int(produce_bar_idx)
    w[gemm_field.NUM_TILES] = 1
    w[gemm_field.TILE_IDS] = _pack_gemm_tile_coord(tile_m, tile_n)
    return w


def _router_topk_inst(*, layer: int, row: int, wait_target: int) -> list[int]:
    w = _zeros()
    w[router_topk_field.OPCODE] = NmcOp.ROUTER_TOPK
    w[router_topk_field.LAYER] = int(layer)
    w[router_topk_field.ROW] = int(row)
    w[router_topk_field.WAIT_TARGET] = int(wait_target)
    return w


def _route_finalize_inst(*, layer: int) -> list[int]:
    w = _zeros()
    w[route_finalize_field.OPCODE] = NmcOp.ROUTE_FINALIZE
    w[route_finalize_field.LAYER] = int(layer)
    return w


def _qkv_inst(
    *,
    layer: int,
    tile_m: int,
    tile_n: int,
    gemm_idx: int,
    needs_rope: bool,
    k_tiles: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
) -> list[int]:
    w = _zeros()
    w[qkv_field.OPCODE] = NmcOp.QKV_PROJ
    w[qkv_field.LAYER] = int(layer)
    w[qkv_field.TILE_M] = int(tile_m)
    w[qkv_field.TILE_N] = int(tile_n)
    w[qkv_field.GEMM_IDX] = int(gemm_idx)
    w[qkv_field.NEEDS_ROPE] = int(bool(needs_rope))
    w[qkv_field.K_TILES] = int(k_tiles)
    w[qkv_field.WAIT_BAR_IDX] = int(wait_bar_idx)
    w[qkv_field.WAIT_TARGET] = int(wait_target)
    w[qkv_field.PRODUCE_BAR_IDX] = int(produce_bar_idx)
    return w


def _attn_decode_inst(
    *,
    layer: int,
    batch_idx: int,
    kv_head_idx: int,
    split_idx: int,
    num_splits: int,
    window_size: int,
    wait_target: int,
) -> list[int]:
    w = _zeros()
    w[attn_field.OPCODE] = NmcOp.ATTN_DECODE
    w[attn_field.LAYER] = int(layer)
    w[attn_field.BATCH_IDX] = int(batch_idx)
    w[attn_field.KV_HEAD_IDX] = int(kv_head_idx)
    w[attn_field.SPLIT_IDX] = int(split_idx)
    w[attn_field.NUM_SPLITS] = int(num_splits)
    w[attn_field.WINDOW_SIZE] = int(window_size)
    w[attn_field.WAIT_TARGET] = int(wait_target)
    w[attn_field.NUM_TILES] = 1
    w[attn_field.TILE_IDS] = _pack_attn_tile(batch_idx, kv_head_idx)
    return w


def _attn_combine_inst(
    *,
    layer: int,
    batch_idx: int,
    kv_head_idx: int,
    num_splits: int,
    wait_bar_idx: int,
    wait_target: int,
    produce_bar_idx: int,
) -> list[int]:
    w = _zeros()
    w[combine_field.OPCODE] = NmcOp.ATTN_COMBINE
    w[combine_field.LAYER] = int(layer)
    w[combine_field.BATCH_IDX] = int(batch_idx)
    w[combine_field.KV_HEAD_IDX] = int(kv_head_idx)
    w[combine_field.NUM_SPLITS] = int(num_splits)
    w[combine_field.WAIT_BAR_IDX] = int(wait_bar_idx)
    w[combine_field.WAIT_TARGET] = int(wait_target)
    w[combine_field.PRODUCE_BAR_IDX] = int(produce_bar_idx)
    return w


def _attn_drain_inst(*, layer: int, queue_offset: int, queue_len: int) -> list[int]:
    """Build a claimer that patches queued ATTN_DECODE words to ``layer``."""
    w = _zeros()
    w[attn_drain_field.OPCODE] = NmcOp.ATTN_DRAIN
    w[attn_drain_field.LAYER] = int(layer)
    w[attn_drain_field.QUEUE_OFFSET] = int(queue_offset)
    w[attn_drain_field.QUEUE_LEN] = int(queue_len)
    return w


def _moe_row_inst(opcode: int, *, layer: int, row: int) -> list[int]:
    w = _zeros()
    w[0] = int(opcode)
    w[1] = int(layer)
    w[2] = int(row)
    return w


def _moe_drain_inst(opcode: int, *, layer: int) -> list[int]:
    w = _zeros()
    w[gemm_field.OPCODE] = int(opcode)
    w[gemm_field.LAYER] = int(layer)
    return w


def _rmsnorm_inst(
    *,
    layer: int,
    row: int,
    wait_ffn_target: int,
    wait_oproj_target: int,
    produce_bar_idx: int,
) -> list[int]:
    """Build an ADD_RMSNORM instruction.

    ``wait_ffn_target``:
      * ``> 0`` -- wait ``bar_ffn_down[layer] >= target`` (dense FFN or
        baseline MoE combine arrivals).
      * ``-1`` -- atomic-TMA MoE-combine sentinel: kernel waits on
        ``bar_route`` then this row's topk fine-grained ``bar_moe_down``
        blocks (no MOE_COMBINE wave). Requires TinyM JIT with
        ``MOE_COMBINE_ATOMIC_TMA=true``.
      * ``0`` -- skip the FFN wait.
    """
    w = _zeros()
    w[rmsnorm_field.OPCODE] = NmcOp.ADD_RMSNORM
    w[rmsnorm_field.LAYER] = int(layer)
    w[rmsnorm_field.ROW] = int(row)
    w[rmsnorm_field.WAIT_FFN_TARGET] = int(wait_ffn_target)
    w[rmsnorm_field.WAIT_OPROJ_TARGET] = int(wait_oproj_target)
    w[rmsnorm_field.PRODUCE_BAR_IDX] = int(produce_bar_idx)
    return w


def _upgate_weight_chunks_for_tiling(tiling: NmcTiling) -> tuple[int, int]:
    """Return the (dense, MoE) fused gate/up interleave chunk for ``tiling``.

    The fused UpGate epilogue stores gate/up halves in ``BN/2``-column tiles, so
    weights must be interleaved with that chunk. The decode kernel derives the
    same chunk from its BN, so this is the value the loaded weight layout MUST
    have been packed with.
    """
    if tiling.upgate.bn % 2 != 0 or tiling.moe_upgate.bn % 2 != 0:
        raise ValueError("fused UpGate BN must be even for gate/up weight packing")
    return tiling.upgate.bn // 2, tiling.moe_upgate.bn // 2


def _validate_nmc_tiling(cfg: Any, tiling: NmcTiling) -> None:
    if int(cfg.hidden_size) % tiling.down.bk != 0:
        raise ValueError("hidden_size must be divisible by down BK")
    if int(cfg.hidden_size) % tiling.upgate.bk != 0:
        raise ValueError("hidden_size must be divisible by UpGate BK")
    if int(cfg.hidden_size) % tiling.moe_upgate.bk != 0:
        raise ValueError("hidden_size must be divisible by MoE UpGate BK")
    if int(cfg.hidden_size) % tiling.oproj.bk != 0:
        raise ValueError("hidden_size must be divisible by OProj BK")
    if int(cfg.hidden_size) % tiling.lmhead.bk != 0:
        raise ValueError("hidden_size must be divisible by LMHead BK")
    if int(cfg.hidden_size) % tiling.router.bk != 0:
        raise ValueError("hidden_size must be divisible by router BK")
    if NMC_EXPERT_DFF % tiling.moe_down.bk != 0:
        raise ValueError("expert FFN size must be divisible by MoE down BK")
    if int(cfg.prefix_dense_intermediate_size) % tiling.down.bk != 0:
        raise ValueError("dense FFN size must be divisible by down BK")
    dense_up_tiles = _div_ceil(2 * int(cfg.prefix_dense_intermediate_size), tiling.upgate.bn)
    if dense_up_tiles % max(1, tiling.down.split_k) != 0:
        raise ValueError("dense fused UpGate output tiles must divide down split-K")
    if int(cfg.num_key_value_heads) % max(1, tiling.oproj.split_k) != 0:
        raise ValueError("KV heads must divide OProj split-K")
    if tiling.upgate.bn != 128 or tiling.upgate.num_cwg != 2:
        raise ValueError("NMC fused UpGate currently requires BN=128 and NUM_CWG=2")
    if tiling.moe_upgate.bn != 128 or tiling.moe_upgate.num_cwg != 2:
        raise ValueError("NMC fused MoE UpGate currently requires BN=128 and NUM_CWG=2")
    if tiling.moe_down.split_k != 1:
        raise ValueError("NMC MoE Down currently requires split_k=1")
    if tiling.moe_down.bm < tiling.moe_upgate.bm or tiling.moe_down.bm % tiling.moe_upgate.bm != 0:
        raise ValueError("NMC fine-grained MoE barriers require MoE Down BM to be a multiple of MoE UpGate BM")


def _nmc_barrier_dims(cfg: Any, tiling: NmcTiling) -> dict[str, int]:
    D = int(cfg.hidden_size)
    dense_dff = int(cfg.prefix_dense_intermediate_size)
    hq = int(cfg.num_attention_heads)
    hkv = int(cfg.num_key_value_heads)
    hd = int(cfg.head_dim)
    # Fused UpGate consumes two halves (gate/up) per op tile. BN=128 produces
    # 64 hidden cols. Down waits on all raw UpGate tasks needed for
    # the row block, not on per-column barrier slots.
    dense_up_tiles = _div_ceil(2 * dense_dff, tiling.upgate.bn)
    return {
        "dense_up_tiles": dense_up_tiles,
        "dense_down_tiles": _div_ceil(D, tiling.down.bn),
        "q_tiles": (hq * hd) // tiling.qkv.bn,
        "kv_tiles": (hkv * hd) // tiling.qkv.bn,
        "d_tiles": _div_ceil(D, tiling.oproj.bn),
        "lm_tiles": _div_ceil(int(cfg.vocab_size), tiling.lmhead.bn),
        "router_tiles": _div_ceil(NMC_NUM_EXPERTS, tiling.router.bn),
        "silu_tiles_per_down_split": dense_up_tiles // max(1, tiling.down.split_k),
        "kv_heads_per_oproj_split": max(1, hkv // max(1, tiling.oproj.split_k)),
        "qkv_wait_per_kv": (hq // hkv) * (hd // tiling.qkv.bn) + 2 * (hd // tiling.qkv.bn),
        "oproj_tiles_total": _div_ceil(D, tiling.oproj.bn) * max(1, tiling.oproj.split_k),
    }


def _decode_sliding_window_left(cfg: Any) -> int:
    """Return the inclusive left radius for a configured total-key window.

    The checkpoint's ``sliding_window=W`` means W visible keys including the
    current token. The instruction ABI stores a FlashAttention-style left
    radius, which covers ``radius + 1`` keys. Zero is reserved for full
    attention, so a one-token sliding window cannot be represented.
    """
    total_keys = int(cfg.sliding_window)
    if total_keys <= 1:
        raise ValueError(
            "decode sliding_window must exceed 1 because zero is the full-attention sentinel"
        )
    return total_keys - 1


def _choose_attn_splits(
    *,
    batch: int,
    hkv: int,
    num_sms: int,
    seq_len: int,
    window_size: int,
    page_block: int,
    max_attn_splits: int,
    min_attn_chunk: int,
) -> int:
    max_attn_splits = max(1, int(max_attn_splits))
    base_tasks = max(1, int(batch) * int(hkv))
    if max_attn_splits <= 1 or base_tasks >= int(num_sms):
        return 1
    if page_block <= 0:
        raise ValueError("page_block must be positive")
    if min_attn_chunk <= 0 or min_attn_chunk % page_block != 0:
        raise ValueError("--min-attn-chunk must be a positive multiple of --page-block")

    effective_tokens = int(seq_len)
    if window_size > 0:
        # FlashAttention window_size=(W, 0) includes the boundary token:
        # [pos - W, pos], so decode attention may cover W + 1 tokens.
        effective_tokens = min(effective_tokens, int(window_size) + 1)
    # H100 tuning: at <=64k tokens, more than 16 splits adds excess overhead.
    if effective_tokens <= ATTN_SPLIT_SHORT_CONTEXT_TOKENS:
        max_attn_splits = min(max_attn_splits, ATTN_SPLIT_CAP_SHORT_CONTEXT)
    total_blocks = max(1, (effective_tokens + page_block - 1) // page_block)
    min_blocks = max(1, min_attn_chunk // page_block)
    max_by_chunk = max(1, total_blocks // min_blocks)
    target_by_sms = max(1, (int(num_sms) + base_tasks - 1) // base_tasks)
    return max(1, min(max_attn_splits, target_by_sms, max_by_chunk))


def _attn_drain_splits_by_row(
    *,
    batch: int,
    hkv: int,
    num_sms: int,
    window_size: int,
    page_block: int,
    max_attn_splits: int,
    min_attn_chunk: int,
    attn_seq_len: int,
    cache_seqlens: Sequence[int] | None,
    oversub_k: int,
    min_splits: int,
) -> list[int]:
    """Choose per-row splits, balancing ragged work across the available SMs.

    Without live lengths, this returns the uniform policy. Otherwise each row's
    KV span is rounded to page blocks and the total is divided into roughly
    ``oversub_k * num_sms`` tasks, never smaller than ``min_attn_chunk``. Long
    rows therefore receive more splits than short rows.

    ``min_splits`` protects callers with an unconditional combine wave.
    """
    if page_block <= 0:
        raise ValueError("page_block must be positive")
    if min_attn_chunk <= 0 or min_attn_chunk % page_block != 0:
        raise ValueError("min_attn_chunk must be a positive multiple of page_block")
    min_splits = max(1, int(min_splits))

    if cache_seqlens is None:
        uniform = _choose_attn_splits(
            batch=batch,
            hkv=hkv,
            num_sms=num_sms,
            seq_len=int(attn_seq_len),
            window_size=window_size,
            page_block=page_block,
            max_attn_splits=max_attn_splits,
            min_attn_chunk=min_attn_chunk,
        )
        return [max(min_splits, uniform)] * batch

    if len(cache_seqlens) != batch:
        raise ValueError("cache_seqlens length must equal batch")

    max_splits = max(1, int(max_attn_splits))
    if max_splits < min_splits:
        raise ValueError(
            f"max_attn_splits={max_splits} < min_splits={min_splits}; "
            f"the drain combine wave requires at least {min_splits} splits per row")
    base_tasks = max(1, int(batch) * int(hkv))
    if base_tasks >= int(num_sms):
        return [min_splits] * batch

    # Include the current token; windowed kinds cap the resulting span at W+1.
    eff: list[int] = []
    for s in cache_seqlens:
        e = int(s) + 1
        if window_size > 0:
            e = min(e, int(window_size) + 1)
        eff.append(max(1, e))
    # Match the short-context cap in _choose_attn_splits.
    if max(eff) <= ATTN_SPLIT_SHORT_CONTEXT_TOKENS:
        max_splits = min(max_splits, ATTN_SPLIT_CAP_SHORT_CONTEXT)
    max_splits = max(max_splits, min_splits)

    min_blocks = max(1, min_attn_chunk // page_block)
    w = [max(1, (e + page_block - 1) // page_block) for e in eff]  # blocks/row
    total_work = max(1, sum(w))
    k = max(1, int(oversub_k))
    # Target k*num_sms tasks without dropping below the chunk floor.
    g_den = max(1, k * int(num_sms))
    g = max(min_blocks, (hkv * total_work + g_den - 1) // g_den)

    out: list[int] = []
    for wr in w:
        splits = (wr + g // 2) // g if g > 0 else 1  # round(wr / g)
        # Cap so each task still covers >= min_attn_chunk tokens.
        max_by_chunk = max(1, (wr + min_blocks - 1) // min_blocks)
        out.append(max(min_splits, min(splits, max_splits, max_by_chunk)))
    return out


def build_attn_drain_queue(
    *,
    cfg: Any,
    batch: int,
    num_sms: int,
    tiling: NmcTiling,
    page_block: int,
    max_attn_splits: int,
    min_attn_chunk: int,
    cache_seqlens: Sequence[int],
    oversub_k: int,
) -> dict[str, Any]:
    """Build the shared full-attention queue from live row lengths.

    Entries use a layer-0 placeholder that each drain claimer patches at runtime.
    Sliding layers use static scheduling and do not consume this queue.
    """
    if len(cache_seqlens) != batch:
        raise ValueError(
            f"cache_seqlens has {len(cache_seqlens)} rows, expected batch={batch}")
    hkv = int(cfg.num_key_value_heads)
    bd = _nmc_barrier_dims(cfg, tiling)
    qkv_wait_per_kv = bd["qkv_wait_per_kv"]

    attn_seq_len = max(int(s) for s in cache_seqlens) + 1
    splits_by_row = _attn_drain_splits_by_row(
        batch=batch,
        hkv=hkv,
        num_sms=num_sms,
        window_size=0,  # full attention only
        page_block=page_block,
        max_attn_splits=max_attn_splits,
        min_attn_chunk=min_attn_chunk,
        attn_seq_len=attn_seq_len,
        cache_seqlens=cache_seqlens,
        oversub_k=oversub_k,
        min_splits=ATTN_DRAIN_MIN_SPLITS,
    )
    words: list[list[int]] = []
    # Preserve static row/head/split order for baseline comparisons.
    for row in range(batch):
        nsplit = splits_by_row[row]
        for kvh in range(hkv):
            for split in range(nsplit):
                words.append(_attn_decode_inst(
                    layer=0,  # placeholder; patched at claim time
                    batch_idx=row,
                    kv_head_idx=kvh,
                    split_idx=split,
                    num_splits=nsplit,
                    window_size=0,
                    wait_target=qkv_wait_per_kv,
                ))
    return {
        "words": words,
        "per_row_splits": splits_by_row,
        "queue_len": len(words),
    }


def _round_robin_waves(
    waves: Sequence[Sequence[Sequence[int]]],
    num_sms: int,
    device: torch.device,
    *,
    ablation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Build on CPU and upload once; one CUDA tensor per instruction makes setup
    # take minutes. Carrying the RR cursor across waves avoids concentrating
    # each wave's waiters on low-numbered SMs.
    # Ablation appends one GRID_SYNC per SM between waves.
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    per_sm: list[list[Sequence[int]]] = [[] for _ in range(num_sms)]
    rr = 0
    for wi, wave in enumerate(waves):
        for inst in wave:
            per_sm[rr % num_sms].append(inst)
            rr += 1
        if ablation and wi + 1 < len(waves):
            sync = _grid_sync_inst()
            for sm in range(num_sms):
                per_sm[sm].append(sync)
    return _serialize_per_sm_rows(per_sm=per_sm, device=device)


def _serialize_per_sm_rows(
    *,
    per_sm: Sequence[Sequence[Sequence[int]]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack host instruction rows into the persistent-kernel schedule ABI."""
    max_inst = max(1, max(len(x) for x in per_sm))
    host_buf = [
        [list(row) for row in rows] + [_zeros() for _ in range(max_inst - len(rows))]
        for rows in per_sm
    ]
    host_cnt = [len(rows) for rows in per_sm]
    buf = torch.tensor(host_buf, device=device, dtype=torch.int32)
    cnt = torch.tensor(host_cnt, device=device, dtype=torch.int32)
    return buf, cnt


def _dependency_affinity_round_robin_waves(
    named_waves: Sequence[tuple[str, int, Sequence[Sequence[int]]]],
    num_sms: int,
    device: torch.device,
    *,
    ablation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply producer affinity while preserving wave order and RR SM quotas.

    Static attention-combine, router-topk, route-finalize, and gather work can
    reuse host-visible producer SMs. Exact RR quotas keep load imbalance out of
    this A/B comparison.

    Dynamic drain producers are unknown to the host and retain plain RR
    placement. Kernel barriers remain the source of correctness.
    """
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")

    total_rows = sum(len(wave) for _, _, wave in named_waves)
    quota_base, quota_extra = divmod(total_rows, num_sms)
    remaining = [
        quota_base + (1 if sm < quota_extra else 0)
        for sm in range(num_sms)
    ]
    per_sm: list[list[Sequence[int]]] = [[] for _ in range(num_sms)]
    rr_cursor = 0

    # Deterministic producer placements used by the targeted affinities.
    attn_decode_sms: dict[tuple[int, int, int], list[int]] = {}
    router_sms: dict[int, list[int]] = {}
    topk_sms: dict[int, list[int]] = {}
    topk_sm_by_row: dict[tuple[int, int], int] = {}

    affinity_stats = {
        "attn_combine": [0, 0],
        "router_topk": [0, 0],
        "route_finalize": [0, 0],
        "moe_gather": [0, 0],
    }
    dynamic_attn_fallbacks = 0

    def unique_sms(sms: Sequence[int]) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for sm in sms:
            sm = int(sm)
            if sm not in seen:
                seen.add(sm)
                out.append(sm)
        return out

    def likely_late_sms(sms: Sequence[int]) -> list[int]:
        # A deeper host instruction list is more likely to reach the current
        # wave later. Ties follow the live RR cursor for deterministic output.
        return sorted(
            unique_sms(sms),
            key=lambda sm: (
                -len(per_sm[sm]),
                (sm - rr_cursor) % num_sms,
            ),
        )

    def choose_sm(
        *,
        preferred_sms: Sequence[int],
        wave_counts: Sequence[int],
    ) -> int:
        nonlocal rr_cursor

        preferred = [
            sm for sm in unique_sms(preferred_sms)
            if 0 <= sm < num_sms and remaining[sm] > 0
        ]
        if preferred:
            # Preserve wave parallelism: avoid stacking consumers on one SM
            # while another preferred producer SM has fewer rows from this wave.
            min_wave_count = min(wave_counts[sm] for sm in preferred)
            sm = next(
                candidate for candidate in preferred
                if wave_counts[candidate] == min_wave_count
            )
        else:
            available = [sm for sm in range(num_sms) if remaining[sm] > 0]
            if not available:
                raise RuntimeError("RR affinity scheduler exhausted all SM quotas")
            min_wave_count = min(wave_counts[sm] for sm in available)
            sm = min(
                (candidate for candidate in available
                 if wave_counts[candidate] == min_wave_count),
                key=lambda candidate: (candidate - rr_cursor) % num_sms,
            )

        remaining[sm] -= 1
        rr_cursor = (sm + 1) % num_sms
        return sm

    for wi, (_label, _group, wave) in enumerate(named_waves):
        wave_counts = [0] * num_sms
        for inst in wave:
            opcode = int(inst[0])
            layer = int(inst[1])
            preferred_sms: list[int] = []
            affinity_kind: str | None = None

            if opcode == NmcOp.ATTN_COMBINE:
                if int(inst[combine_field.NUM_SPLITS]) == ATTN_DYNAMIC_SENTINEL:
                    # Dynamic drain claims erase the task->SM mapping.
                    dynamic_attn_fallbacks += 1
                else:
                    key = (
                        layer,
                        int(inst[combine_field.BATCH_IDX]),
                        int(inst[combine_field.KV_HEAD_IDX]),
                    )
                    preferred_sms = likely_late_sms(attn_decode_sms.get(key, ()))
                    if preferred_sms:
                        affinity_kind = "attn_combine"
            elif opcode == NmcOp.ROUTER_TOPK:
                row = int(inst[router_topk_field.ROW])
                producers = unique_sms(router_sms.get(layer, ()))
                if producers:
                    # Spread rows over router producers first; the rest of the
                    # producer set is a quota-pressure fallback.
                    primary = producers[row % len(producers)]
                    preferred_sms = [primary] + [
                        sm for sm in likely_late_sms(producers) if sm != primary
                    ]
                    affinity_kind = "router_topk"
            elif opcode == NmcOp.ROUTE_FINALIZE:
                preferred_sms = likely_late_sms(topk_sms.get(layer, ()))
                if preferred_sms:
                    affinity_kind = "route_finalize"
            elif opcode == NmcOp.MOE_GATHER:
                row = int(inst[moe_gather_field.ROW])
                primary = topk_sm_by_row.get((layer, row))
                producers = likely_late_sms(topk_sms.get(layer, ()))
                if primary is not None:
                    preferred_sms = [primary] + [
                        sm for sm in producers if sm != primary
                    ]
                    affinity_kind = "moe_gather"

            if affinity_kind is not None:
                affinity_stats[affinity_kind][1] += 1
            sm = choose_sm(
                preferred_sms=preferred_sms,
                wave_counts=wave_counts,
            )
            per_sm[sm].append(inst)
            wave_counts[sm] += 1
            if affinity_kind is not None and sm in preferred_sms:
                affinity_stats[affinity_kind][0] += 1

            if opcode == NmcOp.ATTN_DECODE:
                key = (
                    layer,
                    int(inst[attn_field.BATCH_IDX]),
                    int(inst[attn_field.KV_HEAD_IDX]),
                )
                attn_decode_sms.setdefault(key, []).append(sm)
            elif opcode == NmcOp.ROUTER_GEMM:
                router_sms.setdefault(layer, []).append(sm)
            elif opcode == NmcOp.ROUTER_TOPK:
                row = int(inst[router_topk_field.ROW])
                topk_sms.setdefault(layer, []).append(sm)
                topk_sm_by_row[(layer, row)] = sm

        if ablation and wi + 1 < len(named_waves):
            sync = _grid_sync_inst()
            for sm in range(num_sms):
                per_sm[sm].append(sync)

    if any(count != 0 for count in remaining):
        raise RuntimeError(
            "RR affinity scheduler did not consume the baseline SM quotas: "
            f"remaining={remaining}")

    total_hits = sum(values[0] for values in affinity_stats.values())
    total_requests = sum(values[1] for values in affinity_stats.values())
    breakdown = " ".join(
        f"{name}={hits}/{requests}"
        for name, (hits, requests) in affinity_stats.items()
    )
    counts = [len(rows) for rows in per_sm]
    logger.info(
        "[mk-release] RR dependency affinity: hits=%d/%d %s "
        "dynamic_attn_fallbacks=%d inst_per_sm=%d..%d",
        total_hits, total_requests, breakdown, dynamic_attn_fallbacks,
        min(counts, default=0), max(counts, default=0),
    )
    return _serialize_per_sm_rows(per_sm=per_sm, device=device)


def reorder_waves(
    layer: int,
    named: dict[str, list[list[int]]],
    *order: str,
) -> list[tuple[str, int, list[list[int]]]]:
    """Flatten one layer's named waves into ``(label, group, wave)`` tuples in
    the requested order.

    ``named`` maps a short wave name (``"qkv"``, ``"attn"``, ``"oproj"``, ...) to
    its instruction rows. ``order`` lists those names in the desired
    round-robin/launch sequence; the emitted label is ``f"L{layer}.{name}"`` so
    the downstream round-robin flatten keeps working unchanged.

    A name in ``order`` that is absent from ``named`` (e.g. ``"attn_combine"``
    when attention is not split) is skipped. Conversely, every wave present in
    ``named`` MUST be listed in ``order`` -- otherwise it would be silently
    dropped from the schedule, so that is a hard error.
    """
    unplaced = set(named) - set(order)
    if unplaced:
        raise ValueError(
            f"layer {layer}: waves {sorted(unplaced)} were built but not placed "
            f"in the requested order {list(order)}")
    return [(f"L{layer}.{name}", layer, named[name]) for name in order if name in named]


def _build_nmc_decode_named_waves(
    *,
    cfg: Any,
    batch: int,
    num_sms: int,
    tiling: NmcTiling,
    page_block: int,
    attn_seq_len: int,
    max_attn_splits: int,
    min_attn_chunk: int,
    include_lm_head: bool,
    attn_drain: bool,
    attn_drain_sms: int | None,
    moe_drain_sms: tuple[int, int] | None,
) -> tuple[list[tuple[str, int, list[list[int]]]], int]:
    """Build named dependency waves for the full ``all`` NMC decode schedule.

    Attention split count for STATIC (sliding / drain-off) layers is fixed when
    the decode state is built. This is a simplification: long running decode
    steps may grow past the initial `attn_seq_len`, but the chosen split count
    remains valid and conservative.

    This is the release RR/all path only: every layer is scheduled, in order,
    so there is no partial-layer (`layer_list`) or HEFT machinery here.

    ``attn_drain`` replaces each FULL-attention layer's static per-(row, kvh,
    split) ATTN_DECODE wave with ATTN_DRAIN claimer rows that pull tasks off a
    shared queue. Sliding layers keep the static wave: with window=4096 /
    page_block=64 a sliding row caps at ~65 KV blocks / 8 splits, and every row
    past the window has identical work, so there is nothing to rebalance
    (confirmed by the drain-sms sweep: attn_drain_sms=64,132 matches 132,132).

    The drain queue itself is NOT built here. The schedule stamps
    ``ATTN_DYNAMIC_SENTINEL`` into ATTN_DRAIN.QUEUE_LEN and
    ATTN_COMBINE.NUM_SPLITS/WAIT_TARGET; the per-step host (Python or C++)
    refreshes ``g.attn_queue_len`` / ``g.attn_num_splits`` from live
    ``cache_seqlens`` before each launch. That is what makes one schedule valid
    for any context length and any raggedness (and what lets the server use
    drain under continuous batching).

    ``attn_drain_sms`` is the claimer count for full layers (``None`` =>
    ``num_sms``). ``moe_drain_sms`` is ``(up, down)``; ``None`` means ``num_sms``.
    """
    _validate_nmc_tiling(cfg, tiling)
    bd = _nmc_barrier_dims(cfg, tiling)
    L = int(cfg.num_hidden_layers)
    if L <= 0:
        raise ValueError("cfg.num_hidden_layers must be positive")
    D = int(cfg.hidden_size)
    dense_dff = int(cfg.prefix_dense_intermediate_size)
    hq = int(cfg.num_attention_heads)
    hkv = int(cfg.num_key_value_heads)
    hd = int(cfg.head_dim)
    q_tiles = bd["q_tiles"]
    kv_tiles = bd["kv_tiles"]
    d_tiles = bd["d_tiles"]
    dense_up_tiles = bd["dense_up_tiles"]
    dense_down_k_tiles = dense_dff // tiling.down.bk
    qkv_k_tiles = D // tiling.qkv.bk
    oproj_k_tiles = (hq * hd) // tiling.oproj.bk
    lm_tiles = bd["lm_tiles"]
    router_tiles = bd["router_tiles"]
    down_splits = max(1, tiling.down.split_k)
    oproj_splits = max(1, tiling.oproj.split_k)
    hr = hq // hkv
    qkv_wait_per_kv = bd["qkv_wait_per_kv"]
    oproj_tiles_total = bd["oproj_tiles_total"]
    kv_heads_per_oproj_split = bd["kv_heads_per_oproj_split"]
    silu_tiles_per_down_split = bd["silu_tiles_per_down_split"]

    if attn_drain_sms is not None and not attn_drain:
        raise ValueError("attn_drain_sms requires attn_drain=True")
    if attn_drain and max_attn_splits < ATTN_DRAIN_MIN_SPLITS:
        raise ValueError(
            f"attn_drain requires max_attn_splits >= {ATTN_DRAIN_MIN_SPLITS} "
            f"(got {max_attn_splits}); the unconditional combine wave needs "
            f"S_b >= {ATTN_DRAIN_MIN_SPLITS}")

    waves: list[tuple[str, int, list[list[int]]]] = []
    lm_wait_target = 2 * batch

    for layer in range(L):
        is_sliding = cfg.layer_types[layer] == "sliding_attention"
        force_prefix_rope = (
            cfg.first_k_dense_replace
            and cfg.prefix_dense_sliding_window_pattern == 1
            and layer < cfg.first_k_dense_replace
        )
        needs_rope = bool(is_sliding or force_prefix_rope)
        window_size = _decode_sliding_window_left(cfg) if is_sliding else 0
        is_moe = layer >= cfg.first_k_dense_replace

        # Build named waves, then choose a barrier-safe RR order below.
        # Ordering changes placement only; barriers encode the layer DAG.
        named: dict[str, list[list[int]]] = {}

        # ---- qkv ----
        qkv_wave: list[list[int]] = []
        for tn in range(q_tiles):
            head = (tn * tiling.qkv.bn) // hd
            kvh = head // hr
            qkv_wave.append(_qkv_inst(
                layer=layer,
                tile_m=0,
                tile_n=tn,
                gemm_idx=0,
                needs_rope=needs_rope,
                k_tiles=qkv_k_tiles,
                wait_bar_idx=layer,
                wait_target=0 if layer == 0 else batch,
                produce_bar_idx=layer * hkv + kvh,
            ))
        for gemm_idx in (1, 2):
            for tn in range(kv_tiles):
                kvh = (tn * tiling.qkv.bn) // hd
                qkv_wave.append(_qkv_inst(
                    layer=layer,
                    tile_m=0,
                    tile_n=tn,
                    gemm_idx=gemm_idx,
                    needs_rope=needs_rope,
                    k_tiles=qkv_k_tiles,
                    wait_bar_idx=layer,
                    wait_target=0 if layer == 0 else batch,
                    produce_bar_idx=layer * hkv + kvh,
                ))
        named["qkv"] = qkv_wave

        # ---- router (MoE) / ffn_upgate (dense) ----
        if is_moe:
            router_wave: list[list[int]] = []
            for split in range(max(1, tiling.router.split_k)):
                for tn in range(router_tiles):
                    router_wave.append(_gemm_inst(
                        NmcOp.ROUTER_GEMM,
                        layer=layer,
                        tile_m=0,
                        tile_n=tn,
                        split=split,
                        k_tiles=D // tiling.router.bk,
                        wait_bar_idx=layer,
                        wait_target=0 if layer == 0 else batch,
                        produce_bar_idx=layer,
                    ))
            named["router"] = router_wave
            # Route preparation depends on the router, not attention.
            named["router_topk"] = [
                _router_topk_inst(
                    layer=layer,
                    row=row,
                    wait_target=router_tiles * max(1, tiling.router.split_k))
                for row in range(batch)
            ]
            named["route_finalize"] = [_route_finalize_inst(layer=layer)]
            named["moe_gather"] = [
                _moe_row_inst(NmcOp.MOE_GATHER, layer=layer, row=row)
                for row in range(batch)
            ]
        else:
            named["ffn_upgate"] = [
                _gemm_inst(
                    NmcOp.FFN_UPGATE_ACT,
                    layer=layer,
                    tile_m=0,
                    tile_n=tn,
                    split=0,
                    k_tiles=D // tiling.upgate.bk,
                    wait_bar_idx=layer,
                    wait_target=0 if layer == 0 else batch,
                    produce_bar_idx=layer * down_splits,
                )
                for tn in range(dense_up_tiles)
            ]

        # ---- attn (+ optional combine) ----
        kvh_per_op = kv_heads_per_oproj_split
        # Drain applies to FULL layers only. Sliding keeps the static wave.
        use_drain = bool(attn_drain) and not is_sliding
        if use_drain:
            # Claimer count tunes parallelism; any positive count is correct,
            # but H100 sweeps showed fewer than num_sms is a performance cliff.
            drain_count = num_sms if attn_drain_sms is None else max(1, int(attn_drain_sms))
            logger.debug(
                "L%d.attn: attn_drain full queue_len=sentinel drain_sms=%d",
                layer, drain_count,
            )
            # Only full attention drains, so its shared queue starts at zero.
            named["attn"] = [
                _attn_drain_inst(
                    layer=layer,
                    queue_offset=0,
                    queue_len=ATTN_DYNAMIC_SENTINEL,
                )
                for _ in range(drain_count)
            ]
            # The sentinel makes each combine read its live per-row split count.
            named["attn_combine"] = [
                _attn_combine_inst(
                    layer=layer,
                    batch_idx=row,
                    kv_head_idx=kvh,
                    num_splits=ATTN_DYNAMIC_SENTINEL,
                    wait_bar_idx=layer * (batch * hkv) + row * hkv + kvh,
                    wait_target=ATTN_DYNAMIC_SENTINEL,
                    produce_bar_idx=layer * oproj_splits + (kvh // kvh_per_op),
                )
                for row in range(batch)
                for kvh in range(hkv)
            ]
        else:
            attn_splits = _choose_attn_splits(
                batch=batch,
                hkv=hkv,
                num_sms=num_sms,
                seq_len=int(attn_seq_len or 1),
                window_size=window_size,
                page_block=page_block,
                max_attn_splits=max_attn_splits,
                min_attn_chunk=min_attn_chunk,
            )
            logger.debug("L%d.attn: attn_splits=%d sliding=%s", layer, attn_splits, is_sliding)

            named["attn"] = [
                _attn_decode_inst(
                    layer=layer,
                    batch_idx=row,
                    kv_head_idx=kvh,
                    split_idx=split,
                    num_splits=attn_splits,
                    window_size=window_size,
                    wait_target=qkv_wait_per_kv,
                )
                for row in range(batch)
                for kvh in range(hkv)
                for split in range(attn_splits)
            ]
            if attn_splits > 1:
                named["attn_combine"] = [
                    _attn_combine_inst(
                        layer=layer,
                        batch_idx=row,
                        kv_head_idx=kvh,
                        num_splits=attn_splits,
                        wait_bar_idx=layer * (batch * hkv) + row * hkv + kvh,
                        wait_target=attn_splits,
                        produce_bar_idx=layer * oproj_splits + (kvh // kvh_per_op),
                    )
                    for row in range(batch)
                    for kvh in range(hkv)
                ]

        # ---- ffn_down (dense) / moe drains + combine (MoE) ----
        if not is_moe:
            down_wave: list[list[int]] = []
            for split in range(down_splits):
                for tn in range(d_tiles):
                    down_wave.append(_gemm_inst(
                        NmcOp.FFN_DOWN,
                        layer=layer,
                        tile_m=0,
                        tile_n=tn,
                        split=split,
                        k_tiles=dense_down_k_tiles,
                        wait_bar_idx=layer * down_splits + split,
                        wait_target=silu_tiles_per_down_split,
                        produce_bar_idx=layer,
                    ))
            named["ffn_down"] = down_wave
            ffn_target = len(down_wave)
        else:
            # Dynamic MoE claimers need only a positive up/down count.
            if moe_drain_sms is None:
                up_drain_count = num_sms
                down_drain_count = num_sms
            else:
                up_drain_count = max(1, int(moe_drain_sms[0]))
                down_drain_count = max(1, int(moe_drain_sms[1]))
                logger.debug(
                    "L%d.moe: moe_drain up_sms=%d down_sms=%d",
                    layer, up_drain_count, down_drain_count,
                )
            named["moe_up_drain"] = [
                _moe_drain_inst(NmcOp.MOE_UPGATE_ACT_DRAIN, layer=layer)
                for _ in range(up_drain_count)
            ]
            named["moe_down_drain"] = [
                _moe_drain_inst(NmcOp.MOE_DOWN_DRAIN, layer=layer)
                for _ in range(down_drain_count)
            ]
            if tiling.moe_combine_atomic_tma:
                # Atomic-TMA arm: MoE down scatter-reduces into x_ffn and
                # arrives bar_moe_down; ADD_RMSNORM waits those blocks
                # directly. Skip the MOE_COMBINE wave; sentinel ffn_target=-1.
                ffn_target = -1
            else:
                named["moe_combine"] = [
                    _moe_row_inst(NmcOp.MOE_COMBINE, layer=layer, row=row)
                    for row in range(batch)
                ]
                ffn_target = batch

        # ---- oproj ----
        oproj_wave: list[list[int]] = []
        for split in range(oproj_splits):
            for tn in range(d_tiles):
                oproj_wave.append(_gemm_inst(
                    NmcOp.O_PROJ,
                    layer=layer,
                    tile_m=0,
                    tile_n=tn,
                    split=split,
                    k_tiles=oproj_k_tiles,
                    wait_bar_idx=layer * oproj_splits + split,
                    # One arrival per row and KV head in this O_PROJ split:
                    # waiting on only the head count is correct at BS1 but lets
                    # larger batches start O_PROJ before all rows finish.
                    wait_target=batch * kv_heads_per_oproj_split,
                    produce_bar_idx=layer,
                ))
        named["oproj"] = oproj_wave

        # ---- rmsnorm (layer join) ----
        if layer < L - 1:
            produce_idx = layer + 1
            lm_wait_target = batch
        else:
            produce_idx = layer
            lm_wait_target = 2 * batch
        named["rmsnorm"] = [
            _rmsnorm_inst(
                layer=layer,
                row=row,
                wait_ffn_target=ffn_target,
                wait_oproj_target=oproj_tiles_total,
                produce_bar_idx=produce_idx,
            )
            for row in range(batch)
        ]

        # ---- ordering ----
        # Attention and FFN run in parallel and rejoin at rmsnorm.
        if not is_moe:
            waves += reorder_waves(
                layer, named,
                "qkv", "ffn_upgate", "attn", "attn_combine",
                "ffn_down", "oproj", "rmsnorm")
        else:
            waves += reorder_waves(
                layer, named,
                "qkv", "router", "router_topk", "route_finalize", "moe_gather",
                "attn", "attn_combine",
                "moe_up_drain", "moe_down_drain", "moe_combine",
                "oproj", "rmsnorm")

    if include_lm_head:
        last_layer = L - 1
        waves.append(("lm_head", -1, [
            _gemm_inst(
                NmcOp.LM_HEAD,
                layer=0,
                tile_m=0,
                tile_n=tn,
                split=0,
                k_tiles=D // tiling.lmhead.bk,
                wait_bar_idx=last_layer,
                wait_target=lm_wait_target,
                produce_bar_idx=0,
            )
            for tn in range(lm_tiles)
        ]))
    return waves, L


def _next_power_of_two(n: int) -> int:
    """
    Returns the next power of two strictly greater than n.
    """
    p = 1 << (n - 1).bit_length()
    if p == n:
        p <<= 1
    return p
def _nmc_context_bucket_upper(context_len: int) -> int:
    """Return the smallest ``1K * 2^n`` bucket covering ``context_len``."""
    context_len = int(context_len)
    if context_len <= 0:
        raise ValueError("NMC context length must be positive")
    upper = NMC_CONTEXT_BUCKET_MIN
    while upper < context_len:
        upper *= 2
    return upper


def _nmc_context_bucket_uppers(*, min_context_len: int, max_context_len: int) -> tuple[int, ...]:
    """Enumerate the bucket upper bounds spanning ``[min, max]`` context.

    Starts at the power-of-two bucket covering ``min_context_len`` and advances
    by ``min(next_power_of_two(upper), upper + 4096)`` so early buckets double
    (cheap, few of them) while very long contexts grow linearly in 4K steps.
    Two contexts landing in the same bucket share a compiled schedule, so the
    ladder decides how many distinct schedules a run has to build.
    """
    if min_context_len <= 0 or max_context_len < min_context_len:
        raise ValueError("require 0 < min_context_len <= max_context_len")
    upper = _nmc_context_bucket_upper(min_context_len)
    out: list[int] = []
    while True:
        out.append(upper)
        if upper >= max_context_len:
            return tuple(out)
        upper = min(_next_power_of_two(upper), upper + 4096)


def _nmc_schedule_split_signature(
    *,
    cfg: Any,
    seq_len: int,
    num_sms: int,
    batch: int,
    max_attn_splits: int,
    min_attn_chunk: int,
    page_block: int,
) -> tuple[int, ...]:
    """Attention-split signature of a schedule at ``seq_len``.

    Two context buckets that pick the same split count for every attention
    window (full-attention plus any sliding window) produce byte-identical
    schedules, so this signature is the dedup key: buckets sharing it reuse one
    ``NmcScheduleChunk`` list and one JIT handle.
    """
    windows = [0]
    if any(t == "sliding_attention" for t in getattr(cfg, "layer_types", [])):
        windows.append(_decode_sliding_window_left(cfg))
    return tuple(
        _choose_attn_splits(
            batch=batch,
            hkv=int(cfg.num_key_value_heads),
            num_sms=num_sms,
            seq_len=seq_len,
            window_size=w,
            page_block=page_block,
            max_attn_splits=max_attn_splits,
            min_attn_chunk=min_attn_chunk,
        )
        for w in windows
    )


@dataclass
class NmcScheduleChunk:
    """One named block of round-robin instruction words for the kernel."""

    label: str
    inst_buf: torch.Tensor
    inst_counts: torch.Tensor


@dataclass(frozen=True)
class NmcDecodeVariant:
    """One context bucket mapped to a precomputed RR instruction schedule.

    The attention work queue is NOT owned by the variant. With per-step
    host-built queues the schedule only stamps ``ATTN_DYNAMIC_SENTINEL`` into
    ATTN_DRAIN / ATTN_COMBINE; the state-owned ``attn_queue_words`` /
    ``attn_num_splits`` buffers are refreshed each step from live lengths.
    """

    bucket_upper: int
    split_signature: tuple[int, ...]
    schedule_key: tuple[Any, ...]
    tiling_key: str
    schedule_chunks: list[NmcScheduleChunk]


@dataclass
class NmcDecodeVariantRegistry:
    """Precomputed RR schedules and process-local JIT handles.

    Multiple bucket entries may share the same schedule chunks when their
    attention-split signatures match.  JIT handles are independently
    deduplicated by serialized tiling configuration.  The current release NMC
    tiling is context-independent, so a registry normally owns exactly one JIT
    handle covering every bucket.

    The already-loaded native extension module is passed explicitly to
    ``ensure_jit_handles`` so library ownership remains with the caller.
    """

    variants: list[NmcDecodeVariant]
    tilings: dict[str, NmcTiling]
    jit_handles: dict[str, Any]  # tiling key -> native launch.JitKernel
    batch: int
    num_sms: int
    num_layers: int
    launch_mode: str
    page_block: int
    max_attn_splits: int
    min_attn_chunk: int
    min_context_len: int
    max_context_len: int
    attn_drain: bool
    closed: bool = False

    def select(self, context_len: int) -> NmcDecodeVariant:
        if self.closed:
            raise RuntimeError("NMC decode variant registry is closed")
        context_len = int(context_len)
        if context_len < self.min_context_len or context_len > self.max_context_len:
            raise ValueError(
                f"NMC context length {context_len} is outside registry range "
                f"[{self.min_context_len}, {self.max_context_len}]")
        for variant in self.variants:
            if context_len <= int(variant.bucket_upper):
                return variant
        raise RuntimeError(
            f"NMC registry has no bucket covering context length {context_len}")

    def unique_schedule_variants(self) -> list[NmcDecodeVariant]:
        seen: set[tuple[Any, ...]] = set()
        out: list[NmcDecodeVariant] = []
        for variant in self.variants:
            if variant.schedule_key in seen:
                continue
            seen.add(variant.schedule_key)
            out.append(variant)
        return out

    def ensure_jit_handles(self, *, library: ModuleType) -> int:
        """Compile missing unique tilings; return the number newly created."""
        if self.closed:
            raise RuntimeError("cannot compile a closed NMC decode registry")
        compiled = 0
        for tiling_key, tiling in self.tilings.items():
            if tiling_key in self.jit_handles:
                continue
            self.jit_handles[tiling_key] = jit_compile(
                library=library, batch=self.batch, tiling=tiling,
                enable_profiler=False)
            compiled += 1
        return compiled

    def jit_handle_for(self, variant: NmcDecodeVariant) -> Any:
        handle = self.jit_handles.get(variant.tiling_key)
        if handle is None:
            raise RuntimeError(
                f"NMC JIT kernel for tiling key {variant.tiling_key!r} is not compiled")
        return handle

    def close(self) -> None:
        """Drop this registry's kernels.

        A reference drop, nothing more: a kernel unloads once nothing names
        it, descriptors included.
        """

        if self.closed:
            return
        self.jit_handles.clear()
        self.closed = True


def build_nmc_decode_schedule_chunks(
    *, cfg: Any, batch: int, num_sms: int, device: torch.device, tiling: NmcTiling,
    attn_seq_len: int, page_block: int, max_attn_splits: int, min_attn_chunk: int,
    attn_drain: bool, attn_drain_sms: int | None, moe_drain_sms: tuple[int, int] | None,
    rr_dependency_affinity: bool, ablation: bool,
) -> list[NmcScheduleChunk]:
    """Build the single ``all`` RR schedule chunk for ``attn_seq_len``.

    Release is mode=all only, so this always returns exactly one chunk.

    With ``attn_drain`` the schedule stamps ``ATTN_DYNAMIC_SENTINEL`` into the
    drain/combine fields; the per-step host owns the live queue buffers.
    """
    if num_sms <= 0:
        raise ValueError("num_sms must be positive")
    named_waves, _ = _build_nmc_decode_named_waves(
        cfg=cfg, batch=batch, num_sms=num_sms, tiling=tiling,
        page_block=page_block, attn_seq_len=attn_seq_len,
        max_attn_splits=max_attn_splits, min_attn_chunk=min_attn_chunk,
        include_lm_head=True, attn_drain=attn_drain,
        attn_drain_sms=attn_drain_sms, moe_drain_sms=moe_drain_sms)
    if rr_dependency_affinity:
        inst, counts = _dependency_affinity_round_robin_waves(
            named_waves, num_sms, device, ablation=ablation)
    else:
        inst, counts = _round_robin_waves(
            [wave for _, _, wave in named_waves], num_sms, device,
            ablation=ablation)
    return [NmcScheduleChunk(label="all", inst_buf=inst, inst_counts=counts)]


def build_nmc_rr_decode_variant_registry(
    *, cfg: Any, batch: int, num_sms: int, device: torch.device, tiling: NmcTiling,
    num_layers: int, launch_mode: str, page_block: int, max_attn_splits: int,
    min_attn_chunk: int, min_context_len: int, max_context_len: int,
    attn_drain: bool, attn_drain_sms: int | None,
    moe_drain_sms: tuple[int, int] | None, rr_dependency_affinity: bool,
    ablation: bool,
) -> NmcDecodeVariantRegistry:
    """Precompute and deduplicate the RR schedules covering a context range.

    Schedules are built at each bucket's UPPER bound so a single schedule is
    valid for every context length in the bucket (it never under-splits as the
    KV cache fills).  Buckets whose attention-split signature matches reuse one
    schedule; JIT handles are compiled lazily by ``ensure_jit_handles``.

    With ``attn_drain``, full-attention layers use the dynamic sentinel, so only
    the sliding-window (static) split count affects the baked schedule words.
    The per-step host refreshes the live full-attention queue from
    ``cache_seqlens``; the registry owns no queue.
    """
    if launch_mode != "all":
        raise ValueError("release NMC decode registry supports only mode=all")
    tiling_key = tiling.as_json()
    common_schedule_key = (
        batch, num_sms, num_layers, str(launch_mode),
        page_block, max_attn_splits, min_attn_chunk, tiling_key,
        bool(attn_drain),
        int(attn_drain_sms) if attn_drain_sms is not None else None,
        tuple(moe_drain_sms) if moe_drain_sms is not None else None,
        bool(rr_dependency_affinity),
        bool(ablation),
    )
    schedules: dict[tuple[Any, ...], list[NmcScheduleChunk]] = {}
    variants: list[NmcDecodeVariant] = []
    has_sliding = any(
        t == "sliding_attention" for t in getattr(cfg, "layer_types", []))
    sliding_window = _decode_sliding_window_left(cfg) if has_sliding else 0
    for bucket_upper in _nmc_context_bucket_uppers(
        min_context_len=min_context_len, max_context_len=max_context_len,
    ):
        split_signature = _nmc_schedule_split_signature(
            cfg=cfg, seq_len=bucket_upper, num_sms=num_sms, batch=batch,
            max_attn_splits=max_attn_splits, min_attn_chunk=min_attn_chunk,
            page_block=page_block)
        if attn_drain:
            # Full layers stamp the dynamic sentinel; only the sliding split
            # count changes the baked instruction words. Without sliding layers
            # the drain schedule is identical across every bucket.
            if has_sliding:
                dedup_signature: tuple[Any, ...] = (
                    _choose_attn_splits(
                        batch=batch,
                        hkv=int(cfg.num_key_value_heads),
                        num_sms=num_sms,
                        seq_len=bucket_upper,
                        window_size=sliding_window,
                        page_block=page_block,
                        max_attn_splits=max_attn_splits,
                        min_attn_chunk=min_attn_chunk,
                    ),
                )
            else:
                dedup_signature = ()
        else:
            dedup_signature = split_signature
        schedule_key = (*common_schedule_key, dedup_signature)
        cached = schedules.get(schedule_key)
        if cached is None:
            cached = build_nmc_decode_schedule_chunks(
                cfg=cfg, batch=batch, num_sms=num_sms, device=device, tiling=tiling,
                attn_seq_len=bucket_upper, page_block=page_block,
                max_attn_splits=max_attn_splits, min_attn_chunk=min_attn_chunk,
                attn_drain=attn_drain, attn_drain_sms=attn_drain_sms,
                moe_drain_sms=moe_drain_sms,
                rr_dependency_affinity=rr_dependency_affinity,
                ablation=ablation)
            schedules[schedule_key] = cached
        variants.append(NmcDecodeVariant(
            bucket_upper=bucket_upper, split_signature=split_signature,
            schedule_key=schedule_key, tiling_key=tiling_key,
            schedule_chunks=cached))

    bucket_lines: list[str] = []
    for variant in variants:
        split_labels = [f"full={variant.split_signature[0]}"]
        if len(variant.split_signature) > 1:
            split_labels.append(f"swa={variant.split_signature[1]}")
        line = (f"  BS={batch}, SEQLEN={variant.bucket_upper}\n"
                f"    attn splits: {', '.join(split_labels)}")
        if attn_drain:
            line += "\n    attn drain: full layers (queue rebuilt per step)"
        bucket_lines.append(line)
    logger.info(
        "[mk-release] RR schedule registry: total_buckets=%d "
        "unique_buckets=%d attn_drain=%s dependency_affinity=%s ablation=%s\n"
        "buckets:\n%s",
        len(variants), len(schedules), bool(attn_drain),
        bool(rr_dependency_affinity), bool(ablation), "\n".join(bucket_lines))
    return NmcDecodeVariantRegistry(
        variants=variants, tilings={tiling_key: tiling}, jit_handles={},
        batch=batch, num_sms=num_sms, num_layers=num_layers,
        launch_mode=str(launch_mode), page_block=page_block,
        max_attn_splits=max_attn_splits, min_attn_chunk=min_attn_chunk,
        min_context_len=min_context_len, max_context_len=max_context_len,
        attn_drain=bool(attn_drain))


@dataclass
class NmcDecodeState:
    """Caller-borrowed tensors plus decode-owned scratch and schedules.

    The host split cache avoids H2D queue uploads while row geometry is stable.
    """

    tensors: dict[str, torch.Tensor]
    desc: NmcLaunchDesc
    kv_cache: Any
    decode_registry: NmcDecodeVariantRegistry | None = None
    active_variant: NmcDecodeVariant | None = None
    schedule_chunks: list[NmcScheduleChunk] | None = None
    attn_drain_splits_host: list[int] | None = None
    # Inputs retained for per-step queue rebuilds.
    attn_drain_cfg: Any | None = None
    attn_drain_tiling: NmcTiling | None = None


def _set_nmc_chunk_desc(state: NmcDecodeState, chunk: NmcScheduleChunk) -> None:
    """Point the launch descriptor at ``chunk``'s instruction tensors."""
    state.desc.inst_buf = _ptr(chunk.inst_buf)
    state.desc.num_inst_per_sm = _ptr(chunk.inst_counts)
    state.desc.max_inst = int(chunk.inst_buf.shape[1])


def _activate_nmc_decode_variant(
    state: NmcDecodeState, *, context_len: int,
) -> tuple[NmcDecodeVariant, bool]:
    """Select and, if changed, install the schedule for ``context_len``.

    Deduplicated buckets still update diagnostics; the state-owned attention
    queue is refreshed separately.
    """
    registry = state.decode_registry
    if registry is None:
        raise RuntimeError("NMC decode state has no dynamic context registry")
    variant = registry.select(context_len)
    changed = state.active_variant is None or (
        state.active_variant.schedule_key != variant.schedule_key)
    if changed:
        state.schedule_chunks = variant.schedule_chunks
        state.tensors["inst_buf"] = variant.schedule_chunks[0].inst_buf
        state.tensors["inst_counts"] = variant.schedule_chunks[0].inst_counts
        _set_nmc_chunk_desc(state, variant.schedule_chunks[0])
    state.active_variant = variant
    return variant, changed


def _refresh_attn_drain_queue(
    state: NmcDecodeState, *, cache_seqlens: Sequence[int],
) -> bool:
    """Refresh the drain queue after the KV step; return whether buffers changed.

    ``cache_seqlens`` must contain this step's write positions because attention
    includes the current token. The H2D copy is skipped while splits are stable.
    """
    registry = state.decode_registry
    if registry is None or not registry.attn_drain:
        return False
    if state.attn_drain_cfg is None or state.attn_drain_tiling is None:
        raise RuntimeError("attn_drain state missing cfg/tiling for queue refresh")
    batch = int(state.desc.BS)
    if len(cache_seqlens) != batch:
        raise ValueError(
            f"cache_seqlens has {len(cache_seqlens)} rows, expected batch={batch}")
    built = build_attn_drain_queue(
        cfg=state.attn_drain_cfg,
        batch=batch,
        num_sms=int(registry.num_sms),
        tiling=state.attn_drain_tiling,
        page_block=int(registry.page_block),
        max_attn_splits=int(registry.max_attn_splits),
        min_attn_chunk=int(registry.min_attn_chunk),
        cache_seqlens=cache_seqlens,
        oversub_k=1,
    )
    splits = [int(s) for s in built["per_row_splits"]]
    words = built["words"]
    queue_len = int(built["queue_len"])
    if not words:
        raise RuntimeError("attn_drain produced an empty task queue")
    capacity = int(state.tensors["attn_queue_words"].shape[0])
    if queue_len > capacity:
        raise RuntimeError(
            f"attn drain queue length {queue_len} exceeds capacity {capacity} "
            f"(batch={batch} hkv={state.desc.Hkv} "
            f"max_attn_splits={registry.max_attn_splits})")
    changed = state.attn_drain_splits_host != splits
    if changed:
        # Upload only the live prefix; trailing capacity stays stale but is
        # never claimed (attn_queue_len bounds the claim loop).
        host_words = torch.tensor(words, dtype=torch.int32)
        state.tensors["attn_queue_words"][:queue_len].copy_(
            host_words, non_blocking=True)
        state.tensors["attn_num_splits"].copy_(
            torch.tensor(splits, dtype=torch.int32), non_blocking=True)
        state.attn_drain_splits_host = splits
    # Publish the live length even when the split vector is unchanged.
    state.desc.attn_queue_len = queue_len
    return changed


_RESET_NAMES = (
    "x_attn", "x_ffn", "router_logits", "moe_x", "moe_hidden", "moe_down_out",
    "topk_experts", "topk_scores", "topk_local_slots", "route_row_for_token",
    "routed_token_ids", "routed_scores", "expert_counts", "expert_offsets",
    "moe_up_task_words", "moe_down_task_words", "moe_up_task_count",
    "moe_down_task_count", "moe_up_task_head", "moe_down_task_head", "lm_logits",
    "bar_upgate", "bar_silu", "bar_ffn_down", "bar_qkv", "bar_combine",
    "bar_attn", "bar_oproj", "bar_layer", "bar_router", "bar_topk", "bar_route",
    "bar_gather", "bar_moe_upgate", "bar_moe_down",
    # Claim cursors must reset every step or drains see an already-consumed queue
    # and the combine wait hangs. attn_num_splits is persistent configuration.
    "attn_queue_heads",
)


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon).to(x.dtype) * gamma


def make_decode_state(
    *, library: ModuleType, prefill: Any, weights: Mapping[str, torch.Tensor],
    cfg: Any, num_sms: int, tiling: NmcTiling,
    decode_registry: NmcDecodeVariantRegistry,
) -> NmcDecodeState:
    """Allocate scratch from a sibling prefill's borrowed KV/cache tensors.

    The initial decode schedule is taken from ``decode_registry`` at the first
    decode context length (``prompt_len + 1``); as decode proceeds the caller
    switches schedules through ``_activate_nmc_decode_variant``.
    """
    _validate_nmc_model_abi(cfg)
    device, dtype = prefill.hidden.device, weights["model.embed_tokens.weight"].dtype
    if device.type != "cuda":
        raise ValueError("release native decode requires CUDA tensors")
    if dtype != torch.bfloat16:
        raise ValueError(
            "the release native decode kernel supports only bfloat16 tensors"
        )
    if prefill.hidden.dtype != torch.bfloat16:
        raise ValueError("prefill hidden state must be bfloat16 for native decode")
    batch, layers, d = int(prefill.input_ids.shape[0]), int(cfg.num_hidden_layers), int(cfg.hidden_size)
    if batch != tiling.batch_size:
        raise ValueError("prefill batch and JIT tiling batch differ")
    kv = prefill.kv_cache
    needed_weights = ("W_upgate", "W_down", "W_qkv", "W_oproj", "W_router", "W_moe_upgate", "W_moe_down",
                      "model.embed_tokens.weight", "input_layernorm.weight", "model.norm.weight")
    missing = [name for name in needed_weights if name not in weights]
    if missing:
        raise KeyError(f"prefill decode missing MK-layout weights: {missing}")

    def weight_tensors(name: str) -> list[torch.Tensor]:
        value = weights[name]
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            return [item for item in value if isinstance(item, torch.Tensor)]
        return []

    wrong_weight_dtypes = [
        name
        for name in needed_weights
        if any(tensor.dtype != torch.bfloat16 for tensor in weight_tensors(name))
    ]
    if wrong_weight_dtypes:
        raise ValueError(
            "native decode requires bfloat16 weights; mismatched tensors: "
            + ", ".join(wrong_weight_dtypes)
        )
    invalid_weight_storage = [
        name
        for name in needed_weights
        if any(
            tensor.device != device or not tensor.is_contiguous()
            for tensor in weight_tensors(name)
        )
    ]
    if invalid_weight_storage:
        raise ValueError(
            "native decode weights must be contiguous on the prefill device: "
            + ", ".join(invalid_weight_storage)
        )
    if kv.k_pool.dtype != torch.bfloat16 or kv.v_pool.dtype != torch.bfloat16:
        raise ValueError("native decode requires bfloat16 KV pools")
    # Decode does not repack fused gate/up weights. A loader/tiling chunk
    # mismatch would silently corrupt dense and MoE output.
    upgate_chunk, moe_upgate_chunk = _upgate_weight_chunks_for_tiling(tiling)
    if int(weights.get("_upgate_chunk", upgate_chunk)) != upgate_chunk:
        raise ValueError("loaded dense UpGate weight layout does not match NMC tiling")
    if int(weights.get("_moe_upgate_chunk", moe_upgate_chunk)) != moe_upgate_chunk:
        raise ValueError("loaded MoE UpGate weight layout does not match NMC tiling")
    token = prefill.next_token.reshape(-1).long()
    x_raw = weights["model.embed_tokens.weight"].index_select(0, token).view(batch, d).contiguous()
    x_resid = _rms_norm(x_raw, weights["input_layernorm.weight"][0], float(cfg.rms_norm_eps)).contiguous()
    moe_bm = max(tiling.moe_upgate.bm, tiling.moe_down.bm, batch)
    max_routed = NMC_NUM_EXPERTS * _div_ceil(batch, moe_bm) * moe_bm
    max_tasks = max(
        _max_moe_row_tiles_total(batch, tiling.moe_upgate.bm)
        * _div_ceil(2 * int(cfg.intermediate_size), tiling.moe_upgate.bn)
        * tiling.moe_upgate.split_k,
        _max_moe_row_tiles_total(batch, tiling.moe_down.bm)
        * _div_ceil(d, tiling.moe_down.bn) * tiling.moe_down.split_k,
    )
    initial_variant = decode_registry.select(int(prefill.prompt_len) + 1)
    initial_chunk = initial_variant.schedule_chunks[0]
    inst, counts = initial_chunk.inst_buf, initial_chunk.inst_counts
    t: dict[str, torch.Tensor] = {
        "W_upgate": weights["W_upgate"], "W_down": weights["W_down"], "W_qkv": weights["W_qkv"],
        "W_oproj": weights["W_oproj"], "W_lmhead": weights["model.embed_tokens.weight"].contiguous(),
        "W_router": weights["W_router"], "W_moe_upgate": weights["W_moe_upgate"], "W_moe_down": weights["W_moe_down"],
        "K_pool": kv.k_pool, "V_pool": kv.v_pool, "page_table": kv.page_table,
        "cache_seqlens": kv.cache_seqlens, "row_active": kv.row_active,
        "cos_table": prefill.cos[:, ::2].contiguous(), "sin_table": prefill.sin[:, ::2].contiguous(),
        "inst_buf": inst, "inst_counts": counts, "x_raw": x_raw, "x_resid": x_resid,
        "upgate_scratch": torch.zeros((batch, 2 * int(cfg.prefix_dense_intermediate_size)), device=device, dtype=dtype),
        "silu_out": torch.zeros((batch, int(cfg.prefix_dense_intermediate_size)), device=device, dtype=dtype),
        "q_out": torch.zeros((layers * batch, int(cfg.num_attention_heads) * int(cfg.head_dim)), device=device, dtype=dtype),
        "o_proj_in": torch.zeros((batch, int(cfg.num_attention_heads) * int(cfg.head_dim)), device=device, dtype=dtype),
        "o_partial": torch.zeros((max(1, num_sms), batch, int(cfg.num_attention_heads) * int(cfg.head_dim)), device=device, dtype=dtype),
        "lse_partial": torch.zeros((max(1, num_sms), batch, int(cfg.num_attention_heads)), device=device, dtype=torch.float32),
        "lm_logits": torch.zeros((batch, int(cfg.vocab_size)), device=device, dtype=dtype),
        "rmsnorm_gamma": torch.stack([weights["input_layernorm.weight"][i + 1] if i < layers - 1
                                      else weights["model.norm.weight"] for i in range(layers)]).contiguous(),
        "x_attn": torch.zeros((batch, d), device=device, dtype=dtype),
        "x_ffn": torch.zeros((batch, d), device=device, dtype=dtype),
        "router_logits": torch.zeros((batch, NMC_NUM_EXPERTS), device=device, dtype=dtype),
        "moe_x": torch.zeros((max_routed, d), device=device, dtype=dtype),
        "moe_hidden": torch.zeros((max_routed, int(cfg.intermediate_size)), device=device, dtype=dtype),
        "moe_down_out": torch.zeros((max_routed, d), device=device, dtype=dtype),
    }
    for name, shape, out_dtype in (
        ("topk_experts", (layers, batch, NMC_TOPK), torch.int32),
        ("topk_scores", (layers, batch, NMC_TOPK), dtype),
        ("topk_local_slots", (layers, batch, NMC_TOPK), torch.int32),
        ("route_row_for_token", (layers, batch, NMC_TOPK), torch.int32),
        ("routed_token_ids", (layers, max_routed), torch.int32),
        ("routed_scores", (layers, max_routed), dtype),
        ("expert_counts", (layers, NMC_NUM_EXPERTS), torch.int32),
        ("expert_offsets", (layers, NMC_NUM_EXPERTS + 1), torch.int32),
        ("moe_up_task_words", (layers, max_tasks, MOE_TASK_RECORD_INTS), torch.int32),
        ("moe_down_task_words", (layers, max_tasks, MOE_TASK_RECORD_INTS), torch.int32),
        ("moe_up_task_count", (layers,), torch.int32), ("moe_down_task_count", (layers,), torch.int32),
        ("moe_up_task_head", (layers,), torch.uint32), ("moe_down_task_head", (layers,), torch.uint32),
        # One claim cursor per layer; drain-off keeps a non-null dummy.
        ("attn_queue_heads",
         (layers,) if decode_registry.attn_drain else (1,), torch.uint32),
        ("prof_buf", (1,), torch.uint64),
    ):
        t[name] = torch.zeros(shape, device=device, dtype=out_dtype)
    # Worst-case drain capacity. Refreshes write only the live prefix and publish
    # its length; drain-off keeps non-null dummy buffers.
    hkv = int(cfg.num_key_value_heads)
    if decode_registry.attn_drain:
        queue_capacity = batch * hkv * max(1, int(decode_registry.max_attn_splits))
        t["attn_queue_words"] = torch.zeros(
            (queue_capacity, NMC_INSTRUCTION_WORDS), device=device, dtype=torch.int32)
        t["attn_num_splits"] = torch.zeros((batch,), device=device, dtype=torch.int32)
    else:
        t["attn_queue_words"] = torch.zeros((1,), device=device, dtype=torch.int32)
        t["attn_num_splits"] = torch.zeros((1,), device=device, dtype=torch.int32)
    moe_row_blocks = _moe_row_blocks_per_layer(batch, tiling.moe_down.bm)
    for name, size in {
        "bar_upgate": layers, "bar_silu": layers * tiling.down.split_k, "bar_ffn_down": layers,
        "bar_qkv": layers * int(cfg.num_key_value_heads), "bar_combine": layers * batch * int(cfg.num_key_value_heads),
        "bar_attn": layers * tiling.oproj.split_k, "bar_oproj": layers, "bar_layer": layers,
        "bar_router": layers, "bar_topk": layers, "bar_route": layers, "bar_gather": layers,
        "bar_moe_upgate": layers * moe_row_blocks,
        "bar_moe_down": layers * moe_row_blocks,
    }.items():
        t[name] = torch.zeros((size,), device=device, dtype=torch.uint32)
    desc = _build_launch_desc(
        library=library,
        tensors=t,
        kv=kv,
        cfg=cfg,
        inst=inst,
        counts=counts,
        device=device,
        num_sms=num_sms,
        layers=layers,
        batch=batch,
        d=d,
        max_routed=max_routed,
        max_tasks=max_tasks,
    )
    state = NmcDecodeState(
        tensors=t, desc=desc, kv_cache=kv, decode_registry=decode_registry,
        active_variant=initial_variant, schedule_chunks=initial_variant.schedule_chunks,
        attn_drain_cfg=cfg, attn_drain_tiling=tiling)
    # One ptr map per unique schedule under cwd/dump/bs{BS}/ (bars shared;
    # inst_buf differs). Live barrier values still come from cuda-gdb x/.
    _dump_nmc_device_ptrs_for_registry(
        tensors=t,
        desc=desc,
        tiling=tiling,
        decode_registry=decode_registry,
        moe_row_blocks=moe_row_blocks,
        max_routed=max_routed,
        max_tasks=max_tasks,
    )
    # Seed the drain queue from the prefill lengths so the first launch is
    # valid even before the step loop's first refresh. Harmless when drain is
    # off (_refresh_attn_drain_queue early-returns).
    _refresh_attn_drain_queue(
        state,
        cache_seqlens=[int(v) for v in kv.cache_seqlens[:batch].detach().cpu().tolist()],
    )
    return state


def _reset(state: NmcDecodeState) -> None:
    for name in _RESET_NAMES:
        state.tensors[name].zero_()


def _seed(state: NmcDecodeState, weights: Mapping[str, torch.Tensor], cfg: Any, token: torch.Tensor) -> None:
    x = weights["model.embed_tokens.weight"].index_select(0, token.reshape(-1).long()).view_as(state.tensors["x_raw"])
    state.tensors["x_raw"].copy_(x)
    state.tensors["x_resid"].copy_(_rms_norm(x, weights["input_layernorm.weight"][0], float(cfg.rms_norm_eps)))


def _launch_nmc_decode(
    *, handle: Any, state: NmcDecodeState, device: torch.device,
) -> tuple[float, float]:
    """Launch once through ``handle`` (a native ``launch.JitKernel``).

    Returns ``(host_wall_ms, cuda_event_ms)``.
    """
    started = time.perf_counter()
    cuda_event_ms = float(handle.decode_launch(desc=state.desc))
    torch.cuda.synchronize(device)
    host_wall_ms = (time.perf_counter() - started) * 1000.0
    return host_wall_ms, cuda_event_ms


def _sample(logits: torch.Tensor, temperature: float, gumbel_sampler: bool) -> torch.Tensor:
    """Sample one token per row from ``logits``.

    The ``gumbel_sampler`` path is vLLM's strategy: perturb the scaled logits
    with Gumbel noise and take an argmax, which draws from the softmax without
    materialising probabilities and avoids the CPU-GPU sync ``torch.multinomial``
    forces. See ``random_sample`` in vllm/v1/sample/ops/topk_topp_sampler.py.
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    scaled = logits / temperature
    if gumbel_sampler:
        return torch.argmax(scaled - torch.empty_like(scaled).exponential_().log(), dim=-1)
    return torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(-1)


def _run_nmc_cpp_decode_loop(
    *, library: ModuleType, state: NmcDecodeState, prefill: Any,
    weights: Mapping[str, torch.Tensor], cfg: Any, jit_handle: Any,
    decode_registry: NmcDecodeVariantRegistry, max_new_tokens: int,
    temperature: float, eos_token_id: int | None, sampling_seed: int,
    decode_base_seqlens: Sequence[int],
) -> tuple[list[list[int]], list[float], int]:
    """Drive the entire decode loop in C++ via ``launch.runtime_generate``.

    Column 0 holds the prefill bootstrap, so the runtime executes
    ``max_new_tokens - 1`` steps. KV positions may be ragged, but output columns
    advance in lockstep. This blocking path has no per-step watchdog.
    """
    device = prefill.hidden.device
    batch = int(prefill.input_ids.shape[0])
    max_new = max(1, int(max_new_tokens))
    generated = torch.zeros((batch, max_new), device=device, dtype=torch.int64)
    generated[:, 0] = prefill.next_token.reshape(-1).to(torch.int64)

    # Match the scratch and barrier regions reset by the Python loop.
    zero_regions = [
        library.launch.NmcZeroRegion(
            ptr=int(state.tensors[name].data_ptr()),
            bytes=int(state.tensors[name].numel())
            * int(state.tensors[name].element_size()),
        )
        for name in _RESET_NAMES
    ]

    w_ln0 = weights["input_layernorm.weight"][0].contiguous()
    gen_len = torch.zeros((batch,), device=device, dtype=torch.int32)
    finish = torch.zeros((batch,), device=device, dtype=torch.int32)

    # Convert context buckets to the native runtime representation.
    runtime_variants = []
    for variant in decode_registry.variants:
        if len(variant.schedule_chunks) != 1:
            raise ValueError(
                "the NMC C++ decode runtime requires one schedule chunk per context bucket")
        chunk = variant.schedule_chunks[0]
        entry = library.launch.NmcRuntimeScheduleVariant()
        entry.inst_buf = _ptr(chunk.inst_buf)
        entry.num_inst_per_sm = _ptr(chunk.inst_counts)
        entry.jit_handle = decode_registry.jit_handle_for(variant)
        entry.max_inst = int(chunk.inst_buf.shape[1])
        entry.bucket_upper = int(variant.bucket_upper)
        runtime_variants.append(entry)
    desc = library.launch.NmcRuntimeGenerateDesc()
    desc.launch = state.desc
    desc.w_ln0 = _ptr(w_ln0)
    desc.generated_ids = _ptr(generated)
    desc.zero_regions = zero_regions
    desc.eos_token_ids = [] if eos_token_id is None else [int(eos_token_id)]
    desc.generated_lengths = _ptr(gen_len)
    desc.finish_reasons = _ptr(finish)
    desc.max_new = max_new
    if len(decode_base_seqlens) != batch:
        raise ValueError(
            f"decode_base_seqlens has {len(decode_base_seqlens)} entries for "
            f"batch={batch}")
    # Use one per-row path for uniform and ragged batches; start_pos stays the max.
    desc.start_pos = max(int(v) for v in decode_base_seqlens)
    desc.start_pos_per_row = [int(v) for v in decode_base_seqlens]
    desc.warmup = 0
    desc.vocab_size = int(cfg.vocab_size)
    desc.max_seq_len = int(prefill.cos.shape[0])
    desc.jit_handle = jit_handle
    desc.schedule_variants = runtime_variants
    desc.kv_handle = prefill.kv_cache.handle
    desc.rms_norm_eps = float(cfg.rms_norm_eps)
    desc.temperature = float(temperature)
    desc.sampling_seed = int(sampling_seed)
    desc.timing = 0
    desc.token_callback = 0
    desc.token_callback_context = 0
    desc.attn_drain = 1 if decode_registry.attn_drain else 0
    desc.max_attn_splits = int(decode_registry.max_attn_splits)
    desc.min_attn_chunk = int(decode_registry.min_attn_chunk)

    started = time.perf_counter()
    library.launch.runtime_generate(desc=desc)
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - started) * 1000.0
    # Keep a potentially fresh contiguous buffer alive through the native call.
    del w_ln0

    n_exec = int(desc.executed_steps)
    lengths = gen_len.detach().cpu().tolist()
    rows_cpu = generated.detach().cpu().tolist()
    generated_by_batch: list[list[int]] = []
    for b in range(batch):
        length = int(lengths[b]) if int(lengths[b]) > 0 else (n_exec + 1)
        length = max(0, min(length, max_new))
        row = [int(token) for token in rows_cpu[b][:length]]
        generated_by_batch.append(row)
    steps = max(1, n_exec)
    per_step_ms = total_ms / steps
    return generated_by_batch, [per_step_ms] * n_exec, n_exec


def run_mk_decode_from_prefill(
    *, prefill: Any, weights: Mapping[str, torch.Tensor], cfg: Any, num_sms: int,
    max_new_tokens: int, temperature: float, gumbel_sampler: bool,
    max_attn_splits: int, min_attn_chunk: int, fast: bool, decode_timeout_sec: float,
    cpp_runtime: bool, attn_drain: bool, attn_drain_sms: int | None,
    moe_drain_sms: tuple[int, int] | None, profile_path: str | None,
    profile_max_events: int, rr_dependency_affinity: bool,
    moe_combine_atomic_tma: bool, ablation: bool,
) -> dict[str, Any]:
    """Decode from caller-owned prefill using only RR/all persistent launches.

    The caller must retain ``prefill``, ``weights``, their CUDA tensors, and the
    KV handle until this function returns. ``prefill`` supplies ``input_ids``,
    ``hidden``, ``next_token``, ``cos``, ``sin``, ``prompt_len``, and
    ``kv_cache``; this function owns its temporary decode state and JIT handles.

    ``max_new_tokens`` is the generated buffer width (same as the native
    ``NmcRuntimeGenerateDesc.max_new``).  Column 0 is filled with the prefill
    bootstrap token; both drivers run ``max_new_tokens - 1`` decode steps and
    return the full completion (bootstrap included) so printed text starts at
    the first sampled token.

    Two decode drivers share one setup (schedule registry + JIT handles):

    * Python step loop (default): each step, Python advances the KV cache,
      clears scratch, seeds layer 0, launches the persistent kernel, samples,
      and switches the schedule when the growing context crosses a bucket
      boundary (``_activate_nmc_decode_variant``). With ``attn_drain``, also
      refreshes the full-attention work queue from live ``cache_seqlens``.
    * C++ runtime (``cpp_runtime``): the whole loop runs in
      ``launch.runtime_generate`` via ``_run_nmc_cpp_decode_loop``.

    ``decode_timeout_sec`` bounds each Python-loop step's wall time.  The launch
    is a synchronous C++ call, so this is an *overrun* check: the step still
    runs to completion, then we compare the measured wall time against the
    budget and exit with code 124 if it was exceeded.  It cannot preempt a
    genuinely wedged kernel. The server's C++ watchdog can report such a
    timeout to the host, but it cannot terminate GPU work; a wedged CUDA context
    may still require process restart. A value <= 0 disables the check. The
    ``cpp_runtime`` path is a single blocking native call and does NOT honor
    this timeout.

    ``profile_path`` enables one compact trace for Python-loop step zero. The
    function first warms and measures the normal kernel, then warms the separate
    profiler JIT with a null buffer before running the traced launch. These
    discarded launches reuse the same KV position and are separated by complete
    scratch/barrier resets; their wall time is excluded from ``decode_wall_ms``.
    The trace span covers the first through last recorded operation range, so it
    intentionally excludes leading/trailing kernel control work with no range.

    ``moe_combine_atomic_tma`` selects the A/B arm that removes MOE_COMBINE and
    has MoE down scatter-reduce directly into ``x_ffn`` (TinyM / batch <= 8).

    ``ablation`` inserts a GRID_SYNC on every SM between dependency waves so
    wave N+1 cannot overlap unfinished wave-N producers.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if fast and temperature != 0.0:
        raise ValueError("fast decode is greedy only")
    if attn_drain_sms is not None and not attn_drain:
        raise ValueError("attn_drain_sms requires attn_drain")
    if profile_path is not None:
        if not str(profile_path).strip():
            raise ValueError("profile_path must not be empty")
        if int(profile_max_events) <= 0:
            raise ValueError("profile_max_events must be positive")
        if cpp_runtime:
            raise ValueError("--profile cannot be combined with --cpp-decode-runtime")
    batch = int(prefill.input_ids.shape[0])
    tiling = default_nmc_tiling_for_bs(
        batch=batch, moe_combine_atomic_tma=bool(moe_combine_atomic_tma))
    device = prefill.hidden.device
    driver = "cpp-runtime" if cpp_runtime else "python-loop"
    # Decode steps exclude the prefill bootstrap (column 0).
    decode_steps = max(0, int(max_new_tokens) - 1)
    if profile_path is not None and decode_steps == 0:
        raise ValueError("--profile requires --max-new-tokens >= 2")
    setup_started = time.perf_counter()
    print(
        f"[mk-release] {'FAST' if fast else 'E2E'}: building decode state "
        f"num_sms={num_sms} mode=all driver={driver} "
        f"rr_dependency_affinity={bool(rr_dependency_affinity)} "
        f"ablation={bool(ablation)}",
        flush=True,
    )
    # Precompute and deduplicate context buckets; JIT compilation remains lazy.
    required_min_context = int(prefill.prompt_len) + 1
    required_max_context = int(prefill.prompt_len) + max(1, decode_steps)
    # Snapshot ragged row positions before decode mutates the KV cache.
    decode_base_seqlens = [
        int(v) for v in prefill.kv_cache.cache_seqlens[:batch].detach().cpu().tolist()
    ]
    if max(decode_base_seqlens) != int(prefill.prompt_len):
        raise ValueError(
            f"prefill.prompt_len={int(prefill.prompt_len)} disagrees with "
            f"max(cache_seqlens)={max(decode_base_seqlens)}")
    # Masked fixture rows remain inactive for the whole run.
    decode_row_active = [
        1 if int(v) != 0 else 0
        for v in prefill.kv_cache.row_active[:batch].detach().cpu().tolist()
    ]
    if len(decode_row_active) != batch:
        raise ValueError("row_active length must equal batch")
    if not any(decode_row_active):
        raise ValueError("decode requires at least one active row")
    num_active_rows = sum(decode_row_active)
    library = native.ext()
    decode_registry = build_nmc_rr_decode_variant_registry(
        cfg=cfg, batch=batch, num_sms=num_sms, device=device, tiling=tiling,
        num_layers=int(cfg.num_hidden_layers), launch_mode="all",
        page_block=int(prefill.kv_cache.page_block), max_attn_splits=max_attn_splits,
        min_attn_chunk=min_attn_chunk, min_context_len=required_min_context,
        max_context_len=required_max_context, attn_drain=attn_drain,
        attn_drain_sms=attn_drain_sms, moe_drain_sms=moe_drain_sms,
        rr_dependency_affinity=rr_dependency_affinity, ablation=ablation)
    state = make_decode_state(library=library, prefill=prefill, weights=weights, cfg=cfg, num_sms=num_sms,
                              tiling=tiling, decode_registry=decode_registry)
    setup_ms = (time.perf_counter() - setup_started) * 1000.0
    total_inst = int(state.tensors["inst_counts"].sum().item())
    max_inst_per_sm = int(state.tensors["inst_buf"].shape[1])
    num_buckets = len(decode_registry.variants)
    num_unique_schedules = len(decode_registry.unique_schedule_variants())
    print(
        f"[mk-release] {'FAST' if fast else 'E2E'}: schedule ready "
        f"sms={num_sms} total_inst={total_inst} "
        f"max_inst_per_sm={max_inst_per_sm} buckets={num_buckets} "
        f"unique_schedules={num_unique_schedules} setup={setup_ms:.3f} ms",
        flush=True,
    )
    # Seed each row with the prefill-sampled bootstrap so returned completions
    # include the first token (matches the C++ path, which keeps column 0).
    generated: list[list[int]] = [
        [int(tok)] for tok in prefill.next_token.reshape(-1).detach().cpu().tolist()
    ]
    current = prefill.next_token.reshape(-1).long()
    step_ms: list[float] = []
    profile_handle: Any = None
    profiler: Any = None
    profile_recorded = False
    profile_calibration_wall_ms = 0.0
    profile_baseline_wall_ms: float | None = None
    profile_baseline_cuda_ms: float | None = None
    profiled_wall_ms: float | None = None
    profiled_cuda_ms: float | None = None
    profile_trace_span_ms: float | None = None
    profile_output_path = (
        None if profile_path is None
        else os.path.abspath(os.path.expanduser(str(profile_path))))
    try:
        print("[mk-release] compiling NMC JIT kernel(s)", flush=True)
        jit_started = time.perf_counter()
        new_jits = decode_registry.ensure_jit_handles(library=library)
        jit_ms = (time.perf_counter() - jit_started) * 1000.0
        num_unique_jits = len({id(h) for h in decode_registry.jit_handles.values()})
        print(
            "[mk-release] jit ready "
            f"time={jit_ms:.3f} ms buckets={num_buckets} "
            f"unique_schedules={num_unique_schedules} "
            f"unique_jits={num_unique_jits} new_jits={new_jits}",
            flush=True,
        )
        if profile_output_path is not None:
            profile_parent = os.path.dirname(profile_output_path)
            if profile_parent:
                os.makedirs(profile_parent, exist_ok=True)
            print("[mk-release] compiling profiler-enabled NMC JIT kernel", flush=True)
            profile_jit_started = time.perf_counter()
            profile_handle = jit_compile(
                library=library, batch=batch, tiling=tiling,
                enable_profiler=True)
            profiler = library.launch.Profiler(
                num_sms=num_sms, max_events=int(profile_max_events))
            # create() already initialises the buffer. Reset explicitly anyway:
            # a zeroed buffer is part of this call's contract, not something to
            # inherit from a constructor side effect.
            profiler.init()
            profile_jit_ms = (
                time.perf_counter() - profile_jit_started) * 1000.0
            print(
                f"[mk-release] profiler ready time={profile_jit_ms:.3f} ms "
                f"max_events_per_block={int(profile_max_events)}",
                flush=True,
            )
        stop_on_eos = not fast and getattr(cfg, "eos_token_id", None) is not None
        print(
            f"[mk-release] decode start max_new={max_new_tokens} "
            f"decode_steps={decode_steps} "
            f"start_pos={prefill.prompt_len} "
            f"active_rows={num_active_rows}/{batch} "
            f"stop_on_eos={stop_on_eos} driver={driver}",
            flush=True,
        )
        if cpp_runtime:
            active_variant = state.active_variant
            if active_variant is None:
                raise RuntimeError("decode state has no active schedule variant")
            generated, step_ms, _ = _run_nmc_cpp_decode_loop(
                library=library, state=state, prefill=prefill, weights=weights, cfg=cfg,
                jit_handle=decode_registry.jit_handle_for(active_variant),
                decode_registry=decode_registry, max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token_id=None if (fast or getattr(cfg, "eos_token_id", None) is None)
                else int(cfg.eos_token_id),
                sampling_seed=int(torch.empty((), dtype=torch.int64).random_().item()),
                decode_base_seqlens=decode_base_seqlens)
            # C++ path's step_ms already partitions the full native-loop wall.
            decode_wall_ms = sum(step_ms)
        else:
            # Include host work so Python and C++ decode walls remain comparable.
            # Skip decode if prefill sampled EOS for every active row.
            eos = getattr(cfg, "eos_token_id", None)
            if stop_on_eos and eos is not None and all(
                (not decode_row_active[i]) or row[0] == int(eos)
                for i, row in enumerate(generated)
            ):
                torch.cuda.synchronize(device)
                decode_wall_ms = 0.0
            else:
                torch.cuda.synchronize(device)
                decode_wall_started = time.perf_counter()
                for step in range(decode_steps):
                    # Advance active rows from their own bases; masked rows stay put.
                    # The longest active row selects a conservative context bucket.
                    positions = [
                        (base + step) if decode_row_active[i] else base
                        for i, base in enumerate(decode_base_seqlens)
                    ]
                    pos = max(
                        (positions[i] for i in range(batch) if decode_row_active[i]),
                        default=0,
                    )
                    active_variant, _ = _activate_nmc_decode_variant(
                        state,
                        context_len=pos + 1,
                    )
                    handle = decode_registry.jit_handle_for(active_variant)
                    state.kv_cache.step_decode_positions(positions, decode_row_active)
                    # Refresh the full-attention drain queue from the just-written
                    # cache_seqlens BEFORE reset/launch so claimers see this step's
                    # split geometry. No-op when drain is off.
                    _refresh_attn_drain_queue(state, cache_seqlens=positions)
                    _reset(state)
                    _seed(state, weights, cfg, current)
                    use_profile = profiler is not None and step == 0
                    if use_profile:
                        # Calibrate with matching inputs; reset after each warmup.
                        # A null profiler buffer makes instrumented warmup a no-op.
                        calibration_started = time.perf_counter()
                        _launch_nmc_decode(
                            handle=handle, state=state,
                            device=device)
                        _reset(state)
                        _seed(state, weights, cfg, current)
                        (
                            profile_baseline_wall_ms,
                            profile_baseline_cuda_ms,
                        ) = _launch_nmc_decode(
                            handle=handle, state=state,
                            device=device)
                        _reset(state)
                        _seed(state, weights, cfg, current)
                        saved_prof_buf = int(state.desc.prof_buf)
                        state.desc.prof_buf = 0
                        try:
                            _launch_nmc_decode(
                                handle=profile_handle,
                                state=state, device=device)
                        finally:
                            state.desc.prof_buf = saved_prof_buf
                        _reset(state)
                        _seed(state, weights, cfg, current)
                        profile_calibration_wall_ms += (
                            time.perf_counter() - calibration_started) * 1000.0

                        # Warmup used a null pointer, so the buffer remains clean.
                        state.desc.prof_buf = profiler.device_ptr
                    try:
                        elapsed_ms, cuda_event_ms = _launch_nmc_decode(
                            handle=profile_handle if use_profile else handle,
                            state=state, device=device)
                        if use_profile:
                            profiled_wall_ms = elapsed_ms
                            profiled_cuda_ms = cuda_event_ms
                            profile_recorded = True
                    finally:
                        if use_profile:
                            state.desc.prof_buf = _ptr(state.tensors["prof_buf"])
                    step_ms.append(elapsed_ms)
                    if decode_timeout_sec > 0.0 and elapsed_ms > decode_timeout_sec * 1000.0:
                        print(
                            f"[mk-release] decode step {step} took {elapsed_ms:.3f} ms, "
                            f"over the {decode_timeout_sec * 1000.0:.3f} ms budget; exiting 124",
                            flush=True,
                        )
                        raise SystemExit(124)
                    current = _sample(state.tensors["lm_logits"].float(), temperature, gumbel_sampler).long()
                    for row, token in zip(generated, current.detach().cpu().tolist()):
                        row.append(int(token))
                    if stop_on_eos and eos is not None and all(
                        (not decode_row_active[i]) or row[-1] == int(eos)
                        for i, row in enumerate(generated)
                    ):
                        break
                torch.cuda.synchronize(device)
                decode_wall_ms = max(
                    0.0,
                    (time.perf_counter() - decode_wall_started) * 1000.0
                    - profile_calibration_wall_ms,
                )
    finally:
        try:
            if profiler is not None and profile_recorded:
                assert profile_output_path is not None
                profiler.export_to(filename=profile_output_path)
                profile_trace_span_ms = _compact_profile_span_ms(
                    path=profile_output_path)
                assert profile_baseline_wall_ms is not None
                assert profile_baseline_cuda_ms is not None
                assert profiled_wall_ms is not None
                assert profiled_cuda_ms is not None
                trace_ratio = (
                    profile_trace_span_ms / profile_baseline_cuda_ms)
                instrumentation_overhead_pct = (
                    (profiled_cuda_ms / profile_baseline_cuda_ms) - 1.0
                ) * 100.0
                print(
                    f"[mk-release] profiler trace written to {profile_output_path}",
                    flush=True,
                )
                print(
                    "[mk-release] profiler comparison "
                    f"non_profile_cuda={profile_baseline_cuda_ms:.3f} ms "
                    f"profiled_cuda={profiled_cuda_ms:.3f} ms "
                    f"trace_span={profile_trace_span_ms:.3f} ms "
                    f"trace/non_profile={trace_ratio:.4f}x "
                    f"instrumentation_overhead={instrumentation_overhead_pct:+.2f}% "
                    f"(host_wall={profile_baseline_wall_ms:.3f}"
                    f"->{profiled_wall_ms:.3f} ms)",
                    flush=True,
                )
            elif profiler is not None:
                print(
                    "[mk-release] profiler requested, but decode stopped before "
                    "the first traced step; no trace was written",
                    flush=True,
                )
        finally:
            # Explicitly release native objects that a traceback could retain.
            profiler = None
            profile_handle = None
            decode_registry.close()
    executed_steps = len(step_ms)
    # Masked rows produce no billable tokens.
    total_tokens = num_active_rows * executed_steps
    # Lockstep per-request TPOT is wall/steps, not wall/(batch*steps).
    tpot_ms = decode_wall_ms / max(executed_steps, 1)
    tok_s = (1000.0 * total_tokens) / max(decode_wall_ms, float.fromhex("0x1.0p-1022"))
    print(
        f"[mk-release] decode done driver={driver} steps={executed_steps} "
        f"active_rows={num_active_rows}/{batch} "
        f"total={decode_wall_ms:.3f} ms tpot={tpot_ms:.3f} ms ({tok_s:.2f} tok/s)",
        flush=True,
    )
    result = {
        "token_ids": generated[0], "token_ids_by_batch": generated,
        # decode_step_ms: per-step launch+sync (python) or wall/n (cpp).
        # decode_wall_ms: full decode-loop wall for BOTH drivers (fair TPOT).
        "decode_step_ms": step_ms, "decode_wall_ms": decode_wall_ms,
        "setup_ms": setup_ms, "jit_ms": jit_ms, "total_inst": total_inst,
        "max_inst_per_sm": max_inst_per_sm, "decode_driver": driver,
        "num_context_buckets": num_buckets, "num_unique_schedules": num_unique_schedules,
        "rr_dependency_affinity": bool(rr_dependency_affinity),
        "ablation": bool(ablation),
        "prompt_len": int(prefill.prompt_len), "cache_seqlen": int(state.tensors["cache_seqlens"][0].item()),
        "row_active": list(decode_row_active),
        "num_active_rows": int(num_active_rows),
        # Batch-summed decode-loop tokens; excludes the prefill bootstrap.
        "tpot_ms": tpot_ms,
        "throughput_tok_s": tok_s,
        "num_output_tokens": int(total_tokens),
    }
    if profile_output_path is not None:
        result.update({
            "profile_path": profile_output_path,
            "profile_baseline_cuda_ms": profile_baseline_cuda_ms,
            "profiled_cuda_ms": profiled_cuda_ms,
            "profile_baseline_wall_ms": profile_baseline_wall_ms,
            "profiled_wall_ms": profiled_wall_ms,
            "profile_trace_span_ms": profile_trace_span_ms,
        })
    return result
