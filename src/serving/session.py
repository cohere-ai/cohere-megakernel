"""Continuous-batch NMC decode session.

Owns the persistent C++ decode service, a dedicated scheduler thread, paged KV
rows, prefix-aware PyTorch prefill, BS compaction, and request token queues.
It imports only release-bundle modules.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import decode.schedule as decode_schedule
import native
import prefill.torch_prefill as torch_prefill
import torch

LOG = logging.getLogger(__name__)
_GENERATION_STATS_INTERVAL_S = 5.0
_SESSION_THREAD_JOIN_TIMEOUT_S = 30.0
_DECODE_SERVICE_WATCHDOG_EXPIRED = -2
_WATCHDOG_PROCESS_EXIT_CODE = 1


def _terminate_process_after_watchdog(exc: BaseException) -> None:
    """Print the fatal watchdog error and terminate without CUDA cleanup."""
    # A watchdog timeout leaves the offending kernel in flight. Graceful
    # FastAPI/session shutdown would call back into that wedged CUDA context and
    # can hang too, so terminate the whole process from the decode-driver thread.
    print(
        "[mk-release] FATAL: "
        f"{exc}. The CUDA context is not recoverable; "
        f"terminating the server process with exit code "
        f"{_WATCHDOG_PROCESS_EXIT_CODE}.",
        file=sys.stderr,
        flush=True,
    )
    os._exit(_WATCHDOG_PROCESS_EXIT_CODE)


@dataclass
class NmcSessionRequest:
    request_id: str
    prompt_ids: list[int]
    max_new: int
    temperature: float
    top_p: float
    seed: int
    token_queue: queue.Queue[int | None] = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    finish_reason: str | None = None
    error: BaseException | None = None
    emitted: int = 0
    emitted_ids: list[int] = field(default_factory=list)
    suspended: bool = False
    limit_reached: bool = False
    queued_at: float = field(default_factory=time.monotonic)
    _completion_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def emit(self, token_id: int) -> bool:
        with self._completion_lock:
            if self.done.is_set():
                return False
            self.emitted += 1
            self.emitted_ids.append(int(token_id))
            self.token_queue.put(int(token_id))
            return True

    def finish(self, reason: str) -> None:
        with self._completion_lock:
            if self.done.is_set():
                return
            if self.finish_reason is None:
                self.finish_reason = reason
            self.token_queue.put(None)
            self.done.set()

    def fail(self, exc: BaseException) -> None:
        with self._completion_lock:
            if self.done.is_set():
                return
            self.error = exc
            self.token_queue.put(None)
            self.done.set()

    def iter_tokens(self):
        while True:
            token = self.token_queue.get()
            if token is None:
                if self.error is not None:
                    raise self.error
                return
            yield int(token)


@dataclass
class _Slot:
    row: int
    request: NmcSessionRequest | None = None
    start_pos: int = 0
    temperature: float = 0.0
    seed: int = 0

    @property
    def active(self) -> bool:
        return self.request is not None


@dataclass
class _Geometry:
    batch: int
    registry: decode_schedule.NmcDecodeVariantRegistry
    state: decode_schedule.NmcDecodeState
    kv: Any
    generated: torch.Tensor
    d_gen_col: torch.Tensor
    d_temperature: torch.Tensor
    d_seed: torch.Tensor
    desc: decode_schedule.NmcDecodeServiceDesc

    def release_kv(self) -> None:
        """Hand this geometry's KV blocks back to the shared pool.

        BOTH references have to go. ``desc`` owns a share of the handle so that
        a running decode cannot read freed blocks, and geometries live for the
        whole session, so closing only ``kv`` would pin the blocks until the
        session ended.
        """

        self.kv.close()
        self.desc.kv_handle = None


class NmcDecodeSession:
    """NMC-only persistent decode service with pause-safe prefill and compaction."""

    def __init__(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        cfg: torch_prefill.NmcConfig,
        device: torch.device,
        dtype: torch.dtype,
        session_batch: int,
        session_max_new: int,
        kv_reservation_tokens: int,
        max_context: int,
        num_sms: int,
        page_block: int,
        prefill_chunk_size: int,
        frac_vram_utilization: float,
        max_attn_splits: int,
        min_attn_chunk: int,
        batched_prefill: bool,
        batched_prefill_max_batch: int,
        batched_prefill_token_budget: int,
        batched_prefill_coalesce_ms: float,
        eos_token_id: int | None,
        pause_timeout_ms: int,
        attn_drain: bool = True,
        attn_drain_sms: int | None = None,
    ) -> None:
        if session_batch not in decode_schedule.SUPPORTED_BATCH_SIZES:
            raise ValueError(f"unsupported session batch {session_batch}")
        if session_max_new < 1 or kv_reservation_tokens < 1 or max_context < 1:
            raise ValueError(
                "session_max_new, kv_reservation_tokens, and max_context must be positive"
            )
        if (
            session_max_new > max_context
            or kv_reservation_tokens > max_context
            or max_context > 2_147_483_647
        ):
            raise ValueError(
                "session output/reservation sizes must not exceed the int32 "
                "max_context"
            )
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise ValueError("release native decode session requires a CUDA device")
        if dtype != torch.bfloat16:
            raise ValueError("release native decode session supports only bfloat16")
        if num_sms < 1 or page_block < 1 or prefill_chunk_size < 1:
            raise ValueError(
                "num_sms, page_block, and prefill_chunk_size must be positive"
            )
        if (
            max_attn_splits < 1
            or min_attn_chunk < 1
            or min_attn_chunk % page_block
        ):
            raise ValueError(
                "attention splits must be positive and min_attn_chunk must be "
                "a page-block multiple"
            )
        if (
            not math.isfinite(float(frac_vram_utilization))
            or not 0.0 < float(frac_vram_utilization) < 1.0
        ):
            raise ValueError("frac_vram_utilization must be finite and in (0, 1)")
        if attn_drain_sms is not None and not attn_drain:
            raise ValueError("attn_drain_sms requires attn_drain")
        if attn_drain_sms is not None and int(attn_drain_sms) < 1:
            raise ValueError("attn_drain_sms must be >= 1")
        if attn_drain_sms is not None and int(attn_drain_sms) > int(num_sms):
            raise ValueError("attn_drain_sms cannot exceed num_sms")
        if batched_prefill_max_batch < 2:
            raise ValueError("batched_prefill_max_batch must be >= 2")
        if batched_prefill_token_budget < 2:
            raise ValueError("batched_prefill_token_budget must be >= 2")
        if (
            not math.isfinite(float(batched_prefill_coalesce_ms))
            or batched_prefill_coalesce_ms < 0.0
        ):
            raise ValueError(
                "batched_prefill_coalesce_ms must be finite and non-negative"
            )
        if pause_timeout_ms < 0:
            raise ValueError("pause_timeout_ms must be non-negative")
        if eos_token_id is not None and not 0 <= int(eos_token_id) < int(cfg.vocab_size):
            raise ValueError("eos_token_id is outside the model vocabulary")
        self.weights, self.cfg = weights, cfg
        self.device, self.dtype = resolved_device, dtype
        self.session_batch, self.session_max_new = int(session_batch), int(session_max_new)
        self.kv_reservation_tokens = int(kv_reservation_tokens)
        self.max_context, self.num_sms = int(max_context), int(num_sms)
        self.page_block, self.prefill_chunk_size = int(page_block), int(prefill_chunk_size)
        self.frac_vram_utilization = float(frac_vram_utilization)
        self.max_attn_splits, self.min_attn_chunk = int(max_attn_splits), int(min_attn_chunk)
        self.batched_prefill = bool(batched_prefill) and self.session_batch >= 2
        self.batched_prefill_max_batch = min(
            int(batched_prefill_max_batch),
            self.session_batch,
        )
        self.batched_prefill_token_budget = int(batched_prefill_token_budget)
        self.batched_prefill_coalesce_s = (
            float(batched_prefill_coalesce_ms) / 1000.0
        )
        self.attn_drain = bool(attn_drain)
        self.attn_drain_sms = None if attn_drain_sms is None else int(attn_drain_sms)
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.pause_timeout_ms = int(pause_timeout_ms)
        # The process is already bound to a build (and to its debug setting) by
        # the entry point; a session never chooses one.
        self.library = native.ext()
        self.lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.wake = threading.Event()
        self.pending: collections.deque[NmcSessionRequest] = collections.deque()
        self.stopping = False
        self.closed = False
        self.service_error: str | None = None
        self._stats_report_started = time.monotonic()
        self._prompt_tokens_since_report = 0
        self._generated_tokens_since_report = 0
        self._prefix_cacheable_tokens = 0
        self._prefix_cached_tokens = 0
        self.cos, self.sin = torch_prefill.build_rope(self.max_context, cfg, self.device)
        self.w_ln0 = weights["input_layernorm.weight"][0].contiguous()
        # The descriptor carries this object's raw address (a C function
        # pointer plus an opaque context) for the lifetime of the decode
        # service, so it must stay reachable from the session. Dropping it while
        # the service runs would leave the decode thread calling into freed
        # memory. It is the last field in either descriptor that works this way;
        # see the note on NmcDecodeServiceDesc in src/decode/abi.h.
        self.callback = self.library.launch.TokenCallback(self._on_tokens)
        self.geometries = self._build_geometries()
        self.active = self.geometries[self.session_batch]
        self.kv_total_blocks = int(self.active.kv.stats().total)
        self.slots = [_Slot(row=row) for row in range(self.active.batch)]
        try:
            self.service = self.library.service.DecodeService(desc=self.active.desc)
        except BaseException:
            self._close_geometry_resources()
            raise
        self.decode_thread = threading.Thread(target=self._run_service, name="mk-release-decode", daemon=True)
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, name="mk-release-prefill", daemon=True)
        try:
            self.decode_thread.start()
            self.scheduler_thread.start()
        except BaseException:
            # Thread startup can raise outside Exception; native resources still
            # need deterministic teardown before initialization propagates it.
            self.service.signal_stop()
            if self.decode_thread.is_alive():
                self.decode_thread.join(timeout=_SESSION_THREAD_JOIN_TIMEOUT_S)
            self.service.close()
            self._close_geometry_resources()
            raise

    def _close_geometry_resources(self) -> None:
        for geometry in self.geometries.values():
            geometry.release_kv()
            geometry.registry.close()

    def _build_geometries(self) -> dict[int, _Geometry]:
        geometries: dict[int, _Geometry] = {}
        for batch in sorted((b for b in decode_schedule.SUPPORTED_BATCH_SIZES if b <= self.session_batch), reverse=True):
            LOG.info(
                "building decode geometry batch=%d max_context=%d",
                batch,
                self.max_context,
            )
            tiling = decode_schedule.default_nmc_tiling_for_bs(
                batch=batch, moe_combine_atomic_tma=False)
            registry = decode_schedule.build_nmc_rr_decode_variant_registry(
                cfg=self.cfg, batch=batch, num_sms=self.num_sms, device=self.device,
                tiling=tiling, num_layers=self.cfg.num_hidden_layers, launch_mode="all",
                page_block=self.page_block, max_attn_splits=self.max_attn_splits,
                min_attn_chunk=self.min_attn_chunk, min_context_len=1,
                max_context_len=self.max_context,
                # Per-step host-built queue: safe under continuous batching
                # because the C++ service refreshes the queue from live
                # cache_seqlens every step. Sliding layers stay static.
                attn_drain=self.attn_drain,
                attn_drain_sms=self.attn_drain_sms,
                moe_drain_sms=None,
                rr_dependency_affinity=False,
                ablation=False)
            LOG.info(
                "compiling/loading NMC JIT batch=%d buckets=%d unique_schedules=%d",
                batch,
                len(registry.variants),
                len(registry.unique_schedule_variants()),
            )
            compiled = registry.ensure_jit_handles(library=self.library)
            LOG.info(
                "NMC JIT ready batch=%d new_kernels=%d",
                batch,
                compiled,
            )
            kv = torch_prefill.allocate_paged_kv(
                cfg=self.cfg, batch=batch, max_seq=self.max_context,
                page_block=self.page_block, device=self.device, dtype=self.dtype,
                frac_vram_utilization=self.frac_vram_utilization,
                required_max_seq=self.kv_reservation_tokens)
            kv.row_active.zero_()
            kv.cache_seqlens.zero_()
            dummy = torch_prefill.NmcReferencePrefillState(
                input_ids=torch.zeros((batch, 1), device=self.device, dtype=torch.long),
                hidden=torch.zeros((batch, 1, self.cfg.hidden_size), device=self.device, dtype=self.dtype),
                next_token=torch.zeros((batch,), device=self.device, dtype=torch.long),
                cos=self.cos, sin=self.sin, kv_cache=kv, prompt_len=1, max_seq=self.max_context)
            state = decode_schedule.make_decode_state(
                library=self.library,
                prefill=dummy, weights=self.weights, cfg=self.cfg, num_sms=self.num_sms,
                tiling=tiling, decode_registry=registry)
            generated = torch.zeros((batch, self.session_max_new), device=self.device, dtype=torch.int64)
            d_gen_col = torch.zeros((batch,), device=self.device, dtype=torch.int32)
            d_temperature = torch.zeros((batch,), device=self.device, dtype=torch.float32)
            d_seed = torch.zeros((batch,), device=self.device, dtype=torch.int64)
            zero_regions = [
                self.library.launch.NmcZeroRegion(
                    ptr=int(state.tensors[name].data_ptr()),
                    bytes=int(state.tensors[name].numel())
                    * int(state.tensors[name].element_size()),
                )
                for name in decode_schedule._RESET_NAMES
            ]
            variant_entries = []
            for variant in registry.variants:
                entry = self.library.launch.NmcRuntimeScheduleVariant()
                entry.inst_buf = int(variant.schedule_chunks[0].inst_buf.data_ptr())
                entry.num_inst_per_sm = int(variant.schedule_chunks[0].inst_counts.data_ptr())
                entry.jit_handle = registry.jit_handle_for(variant)
                entry.max_inst = int(variant.schedule_chunks[0].inst_buf.shape[1])
                entry.bucket_upper = int(variant.bucket_upper)
                variant_entries.append(entry)
            # Assigned by name: the descriptor's field order lives in
            # src/decode/abi.h, and positional construction would let a change there
            # shift every value here by one slot, silently.
            desc = self.library.launch.NmcDecodeServiceDesc()
            desc.launch = state.desc
            desc.w_ln0 = int(self.w_ln0.data_ptr())
            desc.generated_ids = int(generated.data_ptr())
            desc.zero_regions = zero_regions
            desc.eos_token_ids = [] if self.eos_token_id is None else [int(self.eos_token_id)]
            desc.max_new = self.session_max_new
            desc.vocab_size = self.cfg.vocab_size
            desc.max_seq_len = self.max_context
            desc.kv_handle = kv.handle
            desc.rms_norm_eps = self.cfg.rms_norm_eps
            desc.d_gen_col = int(d_gen_col.data_ptr())
            desc.d_temperature = int(d_temperature.data_ptr())
            desc.d_seed = int(d_seed.data_ptr())
            desc.schedule_variants = variant_entries
            desc.jit_handle = registry.jit_handle_for(registry.variants[0])
            desc.token_callback = self.callback.function_ptr
            desc.token_callback_context = self.callback.context_ptr
            desc.attn_drain = 1 if self.attn_drain else 0
            desc.max_attn_splits = self.max_attn_splits
            desc.min_attn_chunk = self.min_attn_chunk
            geometries[batch] = _Geometry(batch, registry, state, kv, generated, d_gen_col,
                                          d_temperature, d_seed, desc)
            LOG.info("decode geometry ready batch=%d", batch)
        return geometries

    def _on_tokens(self, sampled: Sequence[int], emitted: Sequence[int]) -> None:
        """Native token callback body, invoked once per decode step.

        Runs on the decode thread, which acquires the GIL just for this call,
        so every microsecond here stalls generation for the whole batch. Keep
        it to queueing ids; detokenization and parsing belong to the consumer
        thread.

        Raising aborts the decode loop, so this must not fail on anything
        recoverable. Exceptions need not be swallowed here: the trampoline
        catches them, prints the traceback, and reports the failure.
        """
        for row, token in enumerate(sampled):
            if not emitted[row]:
                continue
            request = self.slots[row].request
            if request is None or request.limit_reached:
                continue
            if not request.emit(token):
                continue
            self._record_generated_tokens(1)
            if request.emitted >= request.max_new:
                request.limit_reached = True
                request.finish_reason = "length"
                self.wake.set()

    def submit(self, prompt_ids: Sequence[int], *, max_new: int, temperature: float, top_p: float) -> NmcSessionRequest:
        ids = [int(value) for value in prompt_ids]
        if not ids:
            raise ValueError("request prompt must contain at least one token")
        if any(token_id < 0 or token_id >= int(self.cfg.vocab_size) for token_id in ids):
            raise ValueError("request prompt contains a token outside the model vocabulary")
        if int(max_new) < 1:
            raise ValueError("max_new must be positive")
        temperature = float(temperature)
        top_p = float(top_p)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(top_p) or top_p != 1.0:
            raise ValueError("release native decode supports only top_p=1.0")
        remaining_context = self.max_context - len(ids)
        budget = min(int(max_new), self.session_max_new, remaining_context)
        if budget < 1:
            raise ValueError("request leaves no generation capacity in release session context")
        request = NmcSessionRequest(
            request_id=f"mk-{uuid.uuid4().hex}", prompt_ids=ids, max_new=budget,
            temperature=temperature, top_p=top_p,
            seed=int(torch.empty((), dtype=torch.int64).random_().item()))
        # A pressure preemption re-prefills prompt + already emitted tokens.
        # Gate the worst request-lifetime footprint now so such a request can
        # never become permanently inadmissible after suspension.
        required_blocks = self._required_blocks_for_prompt(len(ids) + budget)
        if required_blocks > self.kv_total_blocks:
            raise ValueError(
                "request lifetime cannot fit in the configured KV arena "
                f"(required_blocks={required_blocks}, "
                f"total_blocks={self.kv_total_blocks})"
            )
        with self.lock:
            if self.stopping or self.closed:
                raise RuntimeError("decode session is stopping")
            self.pending.append(request)
        self.wake.set()
        return request

    def cancel(self, request: NmcSessionRequest) -> None:
        with self.lock:
            try:
                self.pending.remove(request)
                request.finish("stop")
                return
            except ValueError:
                pass
        request.limit_reached = True
        request.finish_reason = "stop"
        self.wake.set()

    def _run_service(self) -> None:
        rc = int(self.service.run())
        if rc:
            error = decode_schedule._native_error(
                self.library,
                "decode service failed",
            )
            if rc == _DECODE_SERVICE_WATCHDOG_EXPIRED:
                _terminate_process_after_watchdog(error)
                return
            self._fail_session(error)

    def _run_scheduler(self) -> None:
        while not self.stopping:
            self.wake.wait(0.002)
            self.wake.clear()
            try:
                self._tick()
                self._maybe_log_generation_stats()
            except Exception as exc:
                LOG.exception("release decode scheduler failed")
                self._fail_session(exc)
                return

    def _fail_session(self, exc: BaseException) -> None:
        """Stop both workers and fail every unfinished request."""
        with self.lock:
            if self.service_error is None:
                self.service_error = str(exc)
            self.stopping = True
            requests = [
                slot.request for slot in self.slots if slot.request is not None
            ]
            requests.extend(self.pending)
            self.pending.clear()
        for request in requests:
            request.fail(RuntimeError(str(exc)))
        self.wake.set()
        self.service.signal_stop()

    def _record_generated_tokens(self, count: int) -> None:
        with self.stats_lock:
            self._generated_tokens_since_report += int(count)

    def _prefix_hit_rate(self) -> tuple[int, int, float]:
        with self.stats_lock:
            cacheable = self._prefix_cacheable_tokens
            cached = self._prefix_cached_tokens
        return cacheable, cached, 100.0 * cached / max(cacheable, 1)

    def _maybe_log_generation_stats(self) -> None:
        now = time.monotonic()
        with self.stats_lock:
            elapsed = now - self._stats_report_started
            if elapsed < _GENERATION_STATS_INTERVAL_S:
                return
            prompt_tokens = self._prompt_tokens_since_report
            generated_tokens = self._generated_tokens_since_report
            cacheable_prefix_tokens = self._prefix_cacheable_tokens
            cached_prefix_tokens = self._prefix_cached_tokens
        with self.lock:
            waiting = len(self.pending)
        running = sum(slot.active for slot in self.slots)
        # After the engine goes idle, stop spamming zero-throughput lines.
        # Still advance the report window so we don't re-check every scheduler tick.
        if (
            running == 0
            and waiting == 0
            and prompt_tokens == 0
            and generated_tokens == 0
        ):
            with self.stats_lock:
                self._stats_report_started = now
            return
        with self.stats_lock:
            self._prompt_tokens_since_report = 0
            self._generated_tokens_since_report = 0
            self._stats_report_started = now
        kv = self.active.kv.stats()
        kv_usage = 100.0 * int(kv.allocated) / max(int(kv.total), 1)
        prefix_hit_rate = 100.0 * cached_prefix_tokens / max(cacheable_prefix_tokens, 1)
        LOG.info(
            "Avg prompt throughput: %.1f tokens/s, "
            "Avg generation throughput: %.1f tokens/s, "
            "Running: %d reqs, Waiting: %d reqs, GPU KV cache usage: %.1f%%, "
            "Prefix cache token hit rate: %.1f%%",
            prompt_tokens / elapsed,
            generated_tokens / elapsed,
            running,
            waiting,
            kv_usage,
            prefix_hit_rate,
        )

    def _set_slot(
        self,
        *,
        row: int,
        active: bool,
        start_pos: int,
        gen_col: int,
        temperature: float,
        seed: int,
    ) -> None:
        self.service.set_slot(
            row=int(row),
            active=bool(active),
            start_pos=int(start_pos),
            gen_col=int(gen_col),
            temperature=float(temperature),
            seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
        )

    def _state(self):
        # Sized by the service itself, so a geometry switch cannot leave this
        # reading a stale batch.
        active, cols, finish = self.service.get_state()
        return active, cols, finish

    def _required_prefill_blocks(self, request: NmcSessionRequest) -> int:
        """Conservative hybrid-cache peak while prefilling one request."""
        return self._required_blocks_for_prompt(len(request.prompt_ids))

    def _required_blocks_for_prompt(self, prompt_tokens: int) -> int:
        """Conservative hybrid-cache peak for one prompt length."""
        prompt_pages = max(
            1,
            (int(prompt_tokens) + self.page_block - 1) // self.page_block,
        )
        steady_blocks = torch_prefill.estimate_hybrid_kv_blocks(
            layer_types=self.cfg.layer_types,
            row_capacities=[int(prompt_tokens)],
            page_block=self.page_block,
            sw_size=int(self.cfg.sliding_window),
        )
        if int(self.cfg.sliding_window) <= 0:
            return steady_blocks
        sw_pages = torch_prefill.max_sliding_window_pages(
            int(self.cfg.sliding_window),
            self.page_block,
        )
        chunk_pages = torch_prefill.max_sliding_window_pages(
            self.prefill_chunk_size,
            self.page_block,
        )
        # prepare_prefill installs the final SWA window before chunked compute
        # walks forward from the prompt start. Before the next eviction, a
        # sliding layer can also retain one active history window and the pages
        # for the whole current chunk, but never more than the prompt.
        extra_pages_per_sliding_layer = min(
            max(prompt_pages - sw_pages, 0),
            sw_pages + chunk_pages,
        )
        sliding_layers = sum(
            layer_type == "sliding_attention"
            for layer_type in self.cfg.layer_types
        )
        return steady_blocks + sliding_layers * extra_pages_per_sliding_layer

    def _can_admit(self, request: NmcSessionRequest) -> bool:
        # A suspended request stays out of the batch until all remaining live
        # rows drain. Re-admitting it immediately would reclaim the very pages
        # freed to resolve pressure and cause a pressure/preempt thrash loop.
        if request.suspended and any(slot.active for slot in self.slots):
            return False
        return self.active.kv.free_blocks() >= self._required_prefill_blocks(request)

    def _log_prefill_check(self, request: NmcSessionRequest, slot: _Slot) -> None:
        """Log prefill inputs and the prefix-cache rate before this admission."""
        prompt_tokens = len(request.prompt_ids)
        cacheable_tokens = max(
            0,
            ((prompt_tokens - 1) // self.page_block) * self.page_block,
        )
        prior_cacheable, prior_cached, prior_hit_rate = self._prefix_hit_rate()
        prefix = self.library.kv.prefix_cache_stats()
        kv = self.active.kv.stats()
        kv_usage = 100.0 * int(kv.allocated) / max(int(kv.total), 1)
        LOG.info(
            "prefill check request=%s row=%d prompt_tokens=%d "
            "cacheable_prefix_tokens=%d prefix_cache_token_hit_rate=%.1f%% "
            "(cached=%d cacheable=%d) prefix_cache_entries=%d "
            "GPU KV cache usage=%.1f%%",
            request.request_id,
            slot.row,
            prompt_tokens,
            cacheable_tokens,
            prior_hit_rate,
            prior_cached,
            prior_cacheable,
            int(prefix.entries),
            kv_usage,
        )

    def _log_batched_prefill_check(
        self,
        admissions: Sequence[tuple[_Slot, NmcSessionRequest]],
    ) -> None:
        prompt_tokens = sum(
            len(request.prompt_ids) for _slot, request in admissions
        )
        cacheable_tokens = sum(
            max(
                0,
                ((len(request.prompt_ids) - 1) // self.page_block)
                * self.page_block,
            )
            for _slot, request in admissions
        )
        prior_cacheable, prior_cached, prior_hit_rate = self._prefix_hit_rate()
        prefix = self.library.kv.prefix_cache_stats()
        kv = self.active.kv.stats()
        kv_usage = 100.0 * int(kv.allocated) / max(int(kv.total), 1)
        LOG.info(
            "batched prefill check requests=%s rows=%s batch_size=%d "
            "prompt_tokens=%d cacheable_prefix_tokens=%d "
            "prefix_cache_token_hit_rate=%.1f%% (cached=%d cacheable=%d) "
            "prefix_cache_entries=%d GPU KV cache usage=%.1f%%",
            ",".join(request.request_id for _slot, request in admissions),
            ",".join(str(slot.row) for slot, _request in admissions),
            len(admissions),
            prompt_tokens,
            cacheable_tokens,
            prior_hit_rate,
            prior_cached,
            prior_cacheable,
            int(prefix.entries),
            kv_usage,
        )

    def _record_prefill(
        self,
        *,
        request: NmcSessionRequest,
        prompt_tokens: int,
        cached_prefix_tokens: int,
        elapsed_s: float,
    ) -> None:
        cacheable_tokens = max(
            0,
            ((prompt_tokens - 1) // self.page_block) * self.page_block,
        )
        computed_tokens = prompt_tokens - cached_prefix_tokens
        with self.stats_lock:
            self._prompt_tokens_since_report += computed_tokens
            self._prefix_cacheable_tokens += cacheable_tokens
            self._prefix_cached_tokens += cached_prefix_tokens
            cumulative_cacheable = self._prefix_cacheable_tokens
            cumulative_cached = self._prefix_cached_tokens
        LOG.info(
            "prefill completed request=%s prompt_tokens=%d "
            "cached_prefix_tokens=%d computed_tokens=%d "
            "throughput=%.1f tokens/s prefix_cache_token_hit_rate=%.1f%%",
            request.request_id,
            prompt_tokens,
            cached_prefix_tokens,
            computed_tokens,
            computed_tokens / max(elapsed_s, 1e-9),
            100.0 * cumulative_cached / max(cumulative_cacheable, 1),
        )

    def _record_batched_prefill(
        self,
        *,
        requests: Sequence[NmcSessionRequest],
        result: torch_prefill.NmcBatchedPrefillResult,
        elapsed_s: float,
        pause_wait_s: float,
        queue_wait_s: float,
    ) -> None:
        prompt_tokens = sum(result.prompt_lengths)
        cached_prefix_tokens = sum(result.cached_prefix_lengths)
        cacheable_tokens = sum(
            max(
                0,
                ((prompt_length - 1) // self.page_block) * self.page_block,
            )
            for prompt_length in result.prompt_lengths
        )
        with self.stats_lock:
            self._prompt_tokens_since_report += result.computed_tokens
            self._prefix_cacheable_tokens += cacheable_tokens
            self._prefix_cached_tokens += cached_prefix_tokens
            cumulative_cacheable = self._prefix_cacheable_tokens
            cumulative_cached = self._prefix_cached_tokens
        LOG.info(
            "batched prefill completed requests=%s batch_size=%d "
            "prompt_tokens=%d packed_tokens=%d cached_prefix_tokens=%d "
            "queue_wait_ms=%.3f pause_wait_ms=%.3f prefill_ms=%.3f "
            "throughput=%.1f tokens/s prefix_cache_token_hit_rate=%.1f%%",
            ",".join(request.request_id for request in requests),
            len(requests),
            prompt_tokens,
            result.computed_tokens,
            cached_prefix_tokens,
            queue_wait_s * 1000.0,
            pause_wait_s * 1000.0,
            elapsed_s * 1000.0,
            result.computed_tokens / max(elapsed_s, 1e-9),
            100.0 * cumulative_cached / max(cumulative_cacheable, 1),
        )

    def _preempt_one(self, columns: Sequence[int]) -> None:
        """Suspend the largest live KV footprint and retain its client stream."""
        victim = max(
            (slot for slot in self.slots if slot.active),
            key=lambda slot: (
                slot.start_pos + int(columns[slot.row]),
                slot.request.request_id,
            ),
        )
        request = victim.request
        assert request is not None
        self._set_slot(
            row=victim.row,
            active=False,
            start_pos=0,
            gen_col=0,
            temperature=0.0,
            seed=0,
        )
        self.active.kv.free_row(victim.row)
        # The emitted suffix becomes part of the next prefill prompt. `emitted`
        # remains the client-visible token count and must not be reset.
        request.prompt_ids.extend(request.emitted_ids)
        request.emitted_ids.clear()
        request.suspended = True
        request.queued_at = time.monotonic()
        self.pending.append(request)
        self.slots[victim.row] = _Slot(row=victim.row)
        LOG.warning(
            "preempted request=%s emitted=%d reclaimed_row=%d",
            request.request_id,
            request.emitted,
            victim.row,
        )

    def _switch_geometry(self, target_batch: int, live_rows: list[int]) -> None:
        """Compact live rows into a different supported BS while parked."""
        source = self.active
        target = self.geometries[target_batch]
        if len(live_rows) > target.batch:
            raise RuntimeError("target geometry cannot hold live rows")
        _active, columns, _finish = self._state()
        source_rows = [live_rows[index] if index < len(live_rows) else -1
                       for index in range(target.batch)]
        new_handle = source.kv.rebind(
            source_rows, target.kv.page_table, target.kv.cache_seqlens,
            target.kv.row_active)
        try:
            slots = [_Slot(row=row) for row in range(target.batch)]
            carried: list[tuple[int, _Slot, int]] = []
            for dst_row, src_row in enumerate(live_rows):
                target.generated[dst_row].copy_(source.generated[src_row])
                old_slot = self.slots[src_row]
                slots[dst_row] = _Slot(
                    row=dst_row, request=old_slot.request, start_pos=old_slot.start_pos,
                    temperature=old_slot.temperature, seed=old_slot.seed)
                carried.append((dst_row, slots[dst_row], int(columns[src_row])))
            torch.cuda.synchronize(self.device)
        except BaseException:
            # rebind() hands back a handle that no NmcPagedKvCache owns yet.
            # Clearing the local releases its KV blocks -- letting it fall out
            # of scope is not enough, because the traceback keeps this frame
            # (and therefore the handle) alive.
            new_handle = None
            raise
        old_desc_handle = target.desc.kv_handle
        target.desc.kv_handle = new_handle
        try:
            self.service.set_geometry(desc=target.desc)
        except BaseException:
            target.desc.kv_handle = old_desc_handle
            new_handle = None
            raise

        # The service now holds new_handle through target.desc. Commit Python
        # ownership before restoring slots so any later failure closes the
        # geometry it can use.
        #
        # release_kv clears both source.kv and source.desc.kv_handle. Both
        # hold a share of the handle whose rows just moved, and source stays in
        # self.geometries for the rest of the session, so only clearing both
        # returns the blocks the rebind did NOT carry over to the pool.
        source.release_kv()
        if target.kv.handle is not None:
            target.kv.close()
        target.kv.handle = new_handle
        self.active, self.slots = target, slots
        for row, slot, column in carried:
            self._set_slot(
                row=row,
                active=True,
                start_pos=slot.start_pos,
                gen_col=column,
                temperature=slot.temperature,
                seed=slot.seed,
            )

    def _packed_prefill_limit(self) -> int:
        """Maximum full prompt accepted by the no-eviction packed path."""

        return min(
            self.prefill_chunk_size,
            int(self.cfg.sliding_window) - 1,
        )

    def _batch_prefill_eligible(self, request: NmcSessionRequest) -> bool:
        prompt_tokens = len(request.prompt_ids)
        return (
            self.batched_prefill
            and 0 < prompt_tokens <= self._packed_prefill_limit()
            and prompt_tokens <= self.batched_prefill_token_budget
        )

    def _coalesce_pending(self, *, has_done: bool, pressure: bool) -> None:
        """Wait up to the first request's deadline for a short-prompt burst."""

        if (
            not self.batched_prefill
            or self.batched_prefill_coalesce_s <= 0.0
            or has_done
            or pressure
        ):
            return
        available_slots = self.session_batch - sum(
            slot.active for slot in self.slots
        )
        if available_slots < 2:
            return
        while True:
            with self.lock:
                if not self.pending:
                    return
                first = self.pending[0]
                pending_count = len(self.pending)
            if not self._batch_prefill_eligible(first) or not self._can_admit(first):
                return
            capacity = min(
                available_slots,
                self.batched_prefill_max_batch,
            )
            if pending_count >= capacity:
                return
            remaining = (
                first.queued_at
                + self.batched_prefill_coalesce_s
                - time.monotonic()
            )
            if remaining <= 0.0:
                return
            # A new submit wakes this wait. Loop until the original request's
            # fixed deadline so one burst can grow beyond only two requests.
            self.wake.wait(remaining)
            self.wake.clear()

    def _take_prefill_batch(
        self,
        free_slots: Sequence[_Slot],
    ) -> list[tuple[_Slot, NmcSessionRequest]]:
        """Remove the largest eligible FIFO prefix that fits current resources."""

        capacity = min(
            len(free_slots),
            self.batched_prefill_max_batch,
        )
        if capacity < 2:
            return []
        free_blocks = self.active.kv.free_blocks()
        selected: list[NmcSessionRequest] = []
        reserved_blocks = 0
        packed_tokens = 0
        with self.lock:
            for request in self.pending:
                if len(selected) >= capacity:
                    break
                if (
                    not self._batch_prefill_eligible(request)
                    or not self._can_admit(request)
                ):
                    break
                request_blocks = self._required_prefill_blocks(request)
                request_tokens = len(request.prompt_ids)
                if (
                    reserved_blocks + request_blocks > free_blocks
                    or packed_tokens + request_tokens
                    > self.batched_prefill_token_budget
                ):
                    break
                selected.append(request)
                reserved_blocks += request_blocks
                packed_tokens += request_tokens
            if len(selected) < 2:
                return []
            for request in selected:
                popped = self.pending.popleft()
                if popped is not request:
                    raise AssertionError("pending FIFO changed during batch selection")
        return list(zip(free_slots[: len(selected)], selected))

    def _tick(self) -> None:
        active, columns, finishes = self._state()
        pressure = bool(self.service.kv_pressure)
        done = [row for row, slot in enumerate(self.slots) if slot.active and
                (not int(active[row]) or int(finishes[row]) or slot.request.limit_reached)]
        with self.lock:
            need_work = bool(done or self.pending or pressure)
        if not need_work:
            return
        self._coalesce_pending(has_done=bool(done), pressure=pressure)
        pause_started = time.perf_counter()
        self.service.signal_pause(paused=True)
        if not self.service.wait_paused(timeout_ms=self.pause_timeout_ms):
            self.service.signal_pause(paused=False)
            self.service.notify()
            raise RuntimeError("timed out pausing release decode service")
        pause_wait_s = time.perf_counter() - pause_started
        try:
            active, columns, finishes = self._state()
            for row, slot in enumerate(self.slots):
                if not slot.active:
                    continue
                request = slot.request
                if not int(active[row]) or int(finishes[row]) or request.limit_reached:
                    reason = request.finish_reason or ("stop" if int(finishes[row]) == 1 else "length")
                    self._set_slot(
                        row=row,
                        active=False,
                        start_pos=0,
                        gen_col=0,
                        temperature=0.0,
                        seed=0,
                    )
                    self.active.kv.free_row(row)
                    request.finish(reason)
                    self.slots[row] = _Slot(row=row)
            if pressure and any(slot.active for slot in self.slots):
                self._preempt_one(columns)
            live_rows = [slot.row for slot in self.slots if slot.active]
            with self.lock:
                wanted = min(self.session_batch, len(live_rows) + len(self.pending))
            target_batch = next(
                (batch for batch in sorted(self.geometries) if batch >= max(1, wanted)),
                self.session_batch)
            if target_batch != self.active.batch:
                self._switch_geometry(target_batch, live_rows)
            while True:
                free_slots = [slot for slot in self.slots if not slot.active]
                with self.lock:
                    has_pending = bool(self.pending)
                if not has_pending or not free_slots or pressure:
                    break
                admissions = self._take_prefill_batch(free_slots)
                if admissions:
                    self._admit_batch(
                        admissions=admissions,
                        pause_wait_s=pause_wait_s,
                    )
                    continue
                with self.lock:
                    if not self.pending or not self._can_admit(self.pending[0]):
                        break
                    request = self.pending.popleft()
                self._admit(free_slots[0], request)
            if pressure:
                self.service.resume_from_pressure()
        finally:
            self.service.signal_pause(paused=False)
            self.service.notify()

    def _admit(self, slot: _Slot, request: NmcSessionRequest) -> None:
        self._log_prefill_check(request, slot)
        ids = torch.tensor(request.prompt_ids, device=self.device, dtype=torch.long).view(1, -1)
        started = time.perf_counter()
        try:
            token, prompt_len, cached_prefix_len = torch_prefill.nmc_prefill_into_slot(
                session_kv=self.active.kv, row=slot.row, input_ids=ids, weights=self.weights,
                cfg=self.cfg, cos=self.cos, sin=self.sin, prefill_chunk_size=self.prefill_chunk_size,
                temperature=request.temperature, top_p=request.top_p)
        except Exception as exc:  # noqa: BLE001 - fail only this admitted request.
            # Prefill may leave the CUDA context poisoned (async illegal address).
            # free_row then also fails; that must NOT escape to _run_scheduler and
            # kill the whole decode session — fail this request and keep serving.
            try:
                self.active.kv.free_row(slot.row)
            except Exception as free_exc:  # noqa: BLE001
                LOG.exception(
                    "free_row after prefill failure also failed row=%d "
                    "prefill_exc=%s free_exc=%s",
                    slot.row,
                    exc,
                    free_exc,
                )
            request.fail(exc)
            return
        self._record_prefill(
            request=request,
            prompt_tokens=prompt_len,
            cached_prefix_tokens=cached_prefix_len,
            elapsed_s=time.perf_counter() - started,
        )
        self.active.generated[slot.row, 0] = token
        torch.cuda.synchronize(self.device)
        if not request.emit(token):
            self.active.kv.free_row(slot.row)
            return
        self._record_generated_tokens(1)
        if request.emitted >= request.max_new or token == self.eos_token_id:
            self.active.kv.free_row(slot.row)
            request.finish("stop" if token == self.eos_token_id else "length")
            return
        slot.request, slot.start_pos = request, prompt_len
        slot.temperature, slot.seed = request.temperature, request.seed
        request.suspended = False
        self._set_slot(
            row=slot.row,
            active=True,
            start_pos=prompt_len,
            gen_col=0,
            temperature=request.temperature,
            seed=request.seed,
        )

    def _admit_batch(
        self,
        *,
        admissions: Sequence[tuple[_Slot, NmcSessionRequest]],
        pause_wait_s: float,
    ) -> None:
        """Prefill selected rows together and install them after one host sync."""

        slots = [slot for slot, _request in admissions]
        requests = [request for _slot, request in admissions]
        self._log_batched_prefill_check(admissions)
        queue_wait_s = max(
            time.monotonic() - request.queued_at for request in requests
        )
        started = time.perf_counter()
        try:
            result = torch_prefill.nmc_prefill_into_slots(
                session_kv=self.active.kv,
                rows=[slot.row for slot in slots],
                prompt_ids=[request.prompt_ids for request in requests],
                weights=self.weights,
                cfg=self.cfg,
                cos=self.cos,
                sin=self.sin,
                prefill_chunk_size=self.prefill_chunk_size,
                temperatures=[request.temperature for request in requests],
                top_ps=[request.top_p for request in requests],
            )
            row_indices = torch.tensor(
                [slot.row for slot in slots],
                device=self.device,
                dtype=torch.long,
            )
            self.active.generated[:, 0].index_copy_(
                0,
                row_indices,
                result.next_tokens,
            )
            # This D2H copy is the batch's only host synchronization. Stream
            # ordering also guarantees the preceding generated-buffer scatter.
            tokens = [int(token) for token in result.next_tokens.cpu().tolist()]
            prefill_elapsed_s = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - fail only this admitted batch.
            for slot, request in admissions:
                try:
                    self.active.kv.free_row(slot.row)
                except Exception as free_exc:  # noqa: BLE001
                    LOG.exception(
                        "free_row after batched prefill failure also failed "
                        "row=%d prefill_exc=%s free_exc=%s",
                        slot.row,
                        exc,
                        free_exc,
                    )
                request.fail(exc)
            return

        self._record_batched_prefill(
            requests=requests,
            result=result,
            elapsed_s=prefill_elapsed_s,
            pause_wait_s=pause_wait_s,
            queue_wait_s=queue_wait_s,
        )
        for slot, request, token, prompt_len in zip(
            slots,
            requests,
            tokens,
            result.prompt_lengths,
        ):
            if not request.emit(token):
                self.active.kv.free_row(slot.row)
                continue
            self._record_generated_tokens(1)
            if (
                request.limit_reached
                or request.emitted >= request.max_new
                or token == self.eos_token_id
            ):
                self.active.kv.free_row(slot.row)
                request.finish(
                    "stop"
                    if request.limit_reached or token == self.eos_token_id
                    else "length"
                )
                continue
            slot.request = request
            slot.start_pos = prompt_len
            slot.temperature = request.temperature
            slot.seed = request.seed
            request.suspended = False
            self._set_slot(
                row=slot.row,
                active=True,
                start_pos=prompt_len,
                gen_col=0,
                temperature=request.temperature,
                seed=request.seed,
            )

    def warmup_batched_prefill(self) -> None:
        """Compile/autotune the packed path before any production admission."""

        if not self.batched_prefill or self.active.batch < 2:
            return
        safe_limit = self._packed_prefill_limit()
        if safe_limit < 1:
            raise ValueError("batched prefill has no safe prompt length")
        rows = [0, 2] if self.active.batch >= 3 else [0, 1]
        self.library.kv.prefix_cache_clear()
        self.service.signal_pause(paused=True)
        if not self.service.wait_paused(timeout_ms=self.pause_timeout_ms):
            self.service.signal_pause(paused=False)
            self.service.notify()
            raise RuntimeError("timed out pausing decode service for packed warmup")
        try:
            if safe_limit > self.page_block:
                shared_prefix = [1] * self.page_block
                seed_prompt = shared_prefix + [2]
                seed_ids = torch.tensor(
                    seed_prompt,
                    device=self.device,
                    dtype=torch.long,
                ).view(1, -1)
                torch_prefill.nmc_prefill_into_slot(
                    session_kv=self.active.kv,
                    row=rows[0],
                    input_ids=seed_ids,
                    weights=self.weights,
                    cfg=self.cfg,
                    cos=self.cos,
                    sin=self.sin,
                    prefill_chunk_size=self.prefill_chunk_size,
                    temperature=0.0,
                    top_p=1.0,
                )
                self.active.kv.free_row(rows[0])
                first_length = min(safe_limit, 769)
                first_prompt = shared_prefix + [2] * (
                    first_length - self.page_block
                )
            else:
                first_prompt = [1] * safe_limit
            second_length = max(1, min(safe_limit, 257))
            result = torch_prefill.nmc_prefill_into_slots(
                session_kv=self.active.kv,
                rows=rows,
                prompt_ids=(first_prompt, [3] * second_length),
                weights=self.weights,
                cfg=self.cfg,
                cos=self.cos,
                sin=self.sin,
                prefill_chunk_size=self.prefill_chunk_size,
                temperatures=(0.0, 0.0),
                top_ps=(1.0, 1.0),
            )
            result.next_tokens.cpu()
        finally:
            for row in rows:
                self.active.kv.free_row(row)
            self.library.kv.prefix_cache_clear()
            torch.cuda.synchronize(self.device)
            self.service.signal_pause(paused=False)
            self.service.notify()

    def close(self) -> None:
        with self.close_lock:
            if self.closed:
                return
            with self.lock:
                self.stopping = True
                unfinished = [
                    slot.request
                    for slot in self.slots
                    if slot.request is not None
                ]
                unfinished.extend(self.pending)
                self.pending.clear()
            for request in unfinished:
                request.fail(
                    RuntimeError("decode session closed before request completion")
                )
            self.wake.set()
            self.service.signal_stop()
            self.decode_thread.join(timeout=_SESSION_THREAD_JOIN_TIMEOUT_S)
            self.scheduler_thread.join(timeout=_SESSION_THREAD_JOIN_TIMEOUT_S)
            live_threads = [
                thread.name
                for thread in (self.decode_thread, self.scheduler_thread)
                if thread.is_alive()
            ]
            if live_threads:
                # Borrowed CUDA tensors and JIT handles must remain alive while
                # either worker can still touch them. Retain everything so a
                # later close attempt can finish safely.
                raise RuntimeError(
                    "decode session workers did not stop; resources were retained: "
                    + ", ".join(live_threads)
                )
            self.service.close()
            self._close_geometry_resources()
            self.closed = True
