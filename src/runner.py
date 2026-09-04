"""Standalone runner for the megakernel.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

LOG = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "North-Mini-Code-1.0"
DEFAULT_PROMPT = "Write a short Python function that returns the nth Fibonacci number."
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_BATCH_SIZE = 1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_FAKE_PROMPT_LENGTH = 64
DEFAULT_NUM_SMS = 132
DEFAULT_PAGE_BLOCK = 64
# The source fused_norms MoE kernels are tuned for 2048-token chunks. This
# reaches the release 32k-prefill target while still bounding activations.
DEFAULT_PREFILL_CHUNK_SIZE = 2048
DEFAULT_VRAM_UTILIZATION = 0.80
DEFAULT_WARMUP_ITERATIONS = 0
DEFAULT_TIMED_ITERATIONS = 1
DEFAULT_PROFILE_MAX_EVENTS = 65_536
DEFAULT_UPGATE_LAYOUT_CHUNK = 64
DEFAULT_MOE_UPGATE_LAYOUT_CHUNK = 64
MAX_GENERATION_TOKENS = 16_384
NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000
SYNTHETIC_VOCABULARY_SIZE = 256
SYNTHETIC_EOS_TOKEN_ID = 0


@dataclass(frozen=True)
class SamplingParams:
    """Sampling controls for the compatibility runner.

    ``temperature == 0`` means greedy decoding.  The release PyTorch path
    supports only greedy or temperature multinomial sampling; top-p filtering is
    intentionally omitted from this compatibility runner. The native release
    decode ABI supports greedy or temperature-scaled Gumbel-max sampling.
    """

    max_new_tokens: int
    temperature: float

    def validate(self) -> None:
        """Raise ``ValueError`` when a public sampling contract is invalid."""
        if self.max_new_tokens < 1 or self.max_new_tokens > MAX_GENERATION_TOKENS:
            raise ValueError(
                f"max_new_tokens must be in [1, {MAX_GENERATION_TOKENS}]"
            )
        if not math.isfinite(float(self.temperature)) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")


@dataclass(frozen=True)
class GenerationResult:
    """Completed generation owned by the caller after ``NmcRunner.generate``.

    ``token_ids`` contain generated tokens only.  ``finish_reason`` is ``stop``
    for EOS and ``length`` when the requested token budget is exhausted.
    """

    text: str
    token_ids: tuple[int, ...]
    finish_reason: str
    prefill_seconds: float
    decode_seconds: float


@dataclass(frozen=True)
class DecodeMetrics:
    """Decode-phase measurements. Keeps WALL-CLOCK and KERNEL time separate.

    Two very different numbers are reported because they answer different
    questions:

    - ``wall_ms``: wall-clock time of the whole decode loop, measured with a
      host timer and a device sync at both ends. This is the honest end-to-end
      number (it includes host-side sampling, KV stepping, and Python overhead
      between launches). It is NOT the sum of per-step kernel times.
    - ``kernel_ms_total`` / ``kernel_ms_mean``: aggregate of the per-step
      megakernel launch times. Kernel time only -- excludes host-side work.

    ``tokens`` is the ragged-batch decode token count, so ``tok_s`` stays honest
    for batches where some rows hit EOS early.
    """

    wall_ms: float
    kernel_ms_total: float
    steps: int
    tokens: int
    batch_size: int

    @property
    def tok_s(self) -> float:
        # Ragged-batch decode throughput: total emitted tokens / wall time.
        return float(self.tokens) / max(self.wall_ms / 1000.0, 1e-9)

    @property
    def kernel_ms_mean(self) -> float:
        # Mean per-step kernel time (kernel-only TPOT).
        return self.kernel_ms_total / max(1, self.steps)


@dataclass(frozen=True)
class RunMetrics:
    """End-to-end timing summary for a single prefill+decode run.

    ``prefill_wall_ms`` is None when prefill is synthetic (``--fast``), where a
    real prefill throughput is meaningless (the prompt is fake and never runs
    through the model).
    """

    prompt_len: int
    batch_size: int
    decode: DecodeMetrics
    prefill_wall_ms: float | None
    # Per-row prompt tokens served from a cross-request prefix cache. Prefix
    # sharing is a server feature, so this stays 0 here; the field is present
    # to keep one report shape across both front-ends.
    cached_prefix_len: int

    @property
    def prefill_tokens(self) -> int:
        # Tokens ACTUALLY recomputed this run. The whole batch shares one prompt
        # and the first cached_prefix_len tokens of each row came from the prefix
        # cache, so only (prompt_len - cached) * batch tokens were prefilled.
        return max(0, int(self.prompt_len) - int(self.cached_prefix_len)) * int(self.batch_size)

    @property
    def prompt_tokens_total(self) -> int:
        # Full prompt token count including any cached prefix (batch-summed).
        return int(self.prompt_len) * int(self.batch_size)

    @property
    def cache_hit_frac(self) -> float:
        # Fraction of the (per-row) prompt served from cache, in [0, 1].
        if self.prompt_len <= 0:
            return 0.0
        return float(self.cached_prefix_len) / float(self.prompt_len)

    @property
    def prefill_tok_s(self) -> float | None:
        if self.prefill_wall_ms is None:
            return None
        return float(self.prefill_tokens) / max(self.prefill_wall_ms / 1000.0, 1e-9)


@dataclass(frozen=True)
class NativeRunReport:
    """Runner-owned timing and output summary for one native batch launch.

    ``result`` carries the decode scalar fields plus the ``run_metrics`` object
    printed by ``_print_result_summary``.  ``timing_records`` is the per-step
    decode-launch series fed to the per-op timing table.  ``texts`` is None in
    ``--fast`` mode (synthetic logits are not decodable to meaningful text).
    """

    result: dict[str, Any]
    timing_records: list[tuple[str, float]]
    title: str
    texts: tuple[str, ...] | None
    token_ids_by_batch: tuple[tuple[int, ...], ...]
    fast: bool
    batch_size: int


class Tokenizer(Protocol):
    """Minimal tokenizer interface used by the standalone release layer."""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Return model input token IDs for ``text``."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode IDs without mutating tokenizer state."""


class ByteTokenizer:
    """Dependency-free tokenizer for ``--fast`` synthetic benchmarks.

    It is owned by the synthetic backend and is thread-safe because conversion
    has no mutable state.  It must never be used to interpret real NMC output.
    """

    eos_token_id = SYNTHETIC_EOS_TOKEN_ID

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 bytes into non-EOS synthetic token IDs."""
        return [int(byte) + 1 for byte in text.encode("utf-8", "replace")]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode synthetic IDs, dropping the synthetic EOS marker."""
        raw = bytes(
            int(token_id) - 1
            for token_id in token_ids
            if 1 <= int(token_id) < SYNTHETIC_VOCABULARY_SIZE
        )
        return raw.decode("utf-8", "replace")


class TransformersTokenizer:
    """Thread-safe facade around a Hugging Face tokenizer.

    The facade owns no model resources.  Callers must retain the corresponding
    ``NmcRunner`` for the lifetime of requests that use this tokenizer.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_token_id = None if raw_eos is None else int(raw_eos)

    def encode(self, text: str) -> list[int]:
        """Return unpadded input IDs for one text prompt."""
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [int(token_id) for token_id in encoded["input_ids"]]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode generated IDs while retaining model-visible special tokens."""
        return str(self._tokenizer.decode(list(token_ids), skip_special_tokens=False))

    def format_chat(self, messages: Sequence[dict[str, Any]]) -> str:
        """Format validated OpenAI-like messages using the checkpoint template."""
        return str(
            self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        )


class _DecodeBackend(Protocol):
    """Internal decode backend; all calls are serialized by ``NmcRunner``."""

    def next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Return one vocabulary-logit vector for the supplied full sequence."""


class _SyntheticBackend:
    """Deterministic synthetic backend used solely by ``--fast``.

    It exists so launch scripts can exercise the
    broadcast, sampling, reporting, and streaming paths without model weights.
    """

    def next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Return deterministic logits whose winner depends on sequence state."""
        state = sum(int(token_id) for token_id in token_ids) % (
            SYNTHETIC_VOCABULARY_SIZE - 1
        )
        logits = torch.full((SYNTHETIC_VOCABULARY_SIZE,), -32.0)
        next_id = 1 + ((state * 17 + len(token_ids) * 13) % 255)
        logits[next_id] = 32.0
        return logits


class _TransformersBackend:
    """PyTorch reference backend owning one loaded ``AutoModelForCausalLM``."""

    def __init__(self, checkpoint: str, device: torch.device, dtype: torch.dtype) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "real-weight mode requires transformers; install requirements.txt"
            ) from exc
        self.raw_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=False,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=False,
            local_files_only=True,
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        """Run intentionally slow full-context PyTorch reference inference."""
        ids = torch.tensor(list(token_ids), device=self.device, dtype=torch.long).view(1, -1)
        return self.model(input_ids=ids, use_cache=False).logits[0, -1].float()


class NmcRunner:
    """Own model resources and generate NMC text for the release bundle.

    Ownership and threading:
    - The instance owns the model/backend, tokenizer, and its serialization
      lock.  It must remain alive until all requests complete.
    - Public methods are safe to call from many HTTP worker threads; one model
      step executes at a time, which is intentional for the slow reference path.
    - ``close`` releases Python references and must not race a generation.

    Errors: invalid sampling inputs raise ``ValueError``; model loading failures
    raise ``RuntimeError`` or ``FileNotFoundError``. The native MK path is
    handled directly by ``_run_native_benchmark``.
    """

    def __init__(
        self,
        checkpoint: str,
        device_name: str,
        dtype_name: str,
        fast: bool,
    ) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self.checkpoint = checkpoint
        self.device = torch.device(device_name)
        self.dtype = _dtype_from_name(dtype_name)
        self.fast = bool(fast)
        self._backend: _DecodeBackend | None
        if self.fast:
            self.tokenizer = ByteTokenizer()
            self._backend = _SyntheticBackend()
        else:
            backend = _TransformersBackend(checkpoint, self.device, self.dtype)
            self.tokenizer = TransformersTokenizer(backend.raw_tokenizer)
            self._backend = backend

    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult:
        """Generate one completion from ``prompt``.

        The caller owns the returned immutable result.  This method serializes
        the complete slow-generation loop. The release server's continuous
        batching is implemented separately by ``session.NmcDecodeSession``.
        """
        params.validate()
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt tokenization produced no tokens")
        with self._lock:
            self._ensure_open()
            prefill_started = time.monotonic_ns()
            generated: list[int] = []
            logits = self._next_logits(token_ids)
            prefill_seconds = (time.monotonic_ns() - prefill_started) / NANOSECONDS_PER_SECOND
            decode_started = time.monotonic_ns()
            finish_reason = "length"
            for step_index in range(params.max_new_tokens):
                token_id = _sample_token(logits, params.temperature)
                generated.append(token_id)
                if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                if step_index + 1 < params.max_new_tokens:
                    logits = self._next_logits([*token_ids, *generated])
            decode_seconds = (time.monotonic_ns() - decode_started) / NANOSECONDS_PER_SECOND
            return GenerationResult(
                text=self.tokenizer.decode(generated),
                token_ids=tuple(generated),
                finish_reason=finish_reason,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
            )

    def step(self, token_ids: Sequence[int], temperature: float) -> int:
        """Return one next token for a full prompt-plus-output sequence.

        The method only reads the sequence and returns a scalar. It is retained
        for callers of the compatibility runner and raises ``RuntimeError``
        after ``close``.
        """
        if not token_ids:
            raise ValueError("step requires at least one token")
        with self._lock:
            self._ensure_open()
            return _sample_token(self._next_logits(token_ids), temperature)

    def close(self) -> None:
        """Release runner-owned references after all caller-owned work completes."""
        with self._lock:
            self._closed = True
            self._backend = None

    def _next_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        backend = self._backend
        if backend is None:
            raise RuntimeError("runner backend is unavailable")
        return backend.next_logits(token_ids)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("NmcRunner is closed")


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be bf16, fp16, or fp32")


def _sample_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


# Result-dict keys that are bulky generated output or structured objects, not
# scalar values worth dumping on the one-line "metrics = {...}" log.
_NON_SCALAR_RESULT_KEYS = (
    "text",
    "token_ids",
    "texts_by_batch",
    "token_ids_by_batch",
    "run_metrics",
    "decode_metrics",
    "decode_step_ms",
)


def _print_timing_table(records: Sequence[tuple[str, float]], title: str) -> None:
    if not records:
        return
    by_name: dict[str, list[float]] = {}
    for name, ms in records:
        by_name.setdefault(str(name), []).append(float(ms))

    def percentile(vals: list[float], q: float) -> float:
        ordered = sorted(vals)
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
        return ordered[idx]

    rows: list[tuple[float, str, int, float, float, float, float, float]] = []
    total_all = 0.0
    for name, vals in by_name.items():
        total = sum(vals)
        total_all += total
        rows.append((
            total,
            name,
            len(vals),
            total / max(1, len(vals)),
            percentile(vals, 0.5),
            percentile(vals, 0.95),
            min(vals),
            max(vals),
        ))
    rows.sort(reverse=True)
    name_w = max(22, max(len(name) for name in by_name) + 2)
    print(f"\n=== {title} ===", flush=True)
    hdr = (
        f"{'name':<{name_w}} {'count':>6} {'total':>12} {'mean':>12} "
        f"{'median':>12} {'p95':>12} {'min':>12} {'max':>12}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for total, name, n, mean, med, p95, mn, mx in rows:
        print(
            f"{name:<{name_w}} {n:>6d} {total:>12.3f} {mean:>12.4f} "
            f"{med:>12.4f} {p95:>12.4f} {mn:>12.4f} {mx:>12.4f}",
            flush=True,
        )
    print("-" * len(hdr), flush=True)
    print(f"{'TOTAL':<{name_w}} {len(records):>6d} {total_all:>12.3f}", flush=True)


def _print_run_report(metrics: RunMetrics, *, title: str) -> None:
    """Print the compact wall-clock + kernel-time summary for one run.

    This is deliberately separate from ``_print_timing_table`` (the per-step
    kernel-time table). This block is the high-level "what throughput did we
    get" summary.
    """
    d = metrics.decode
    print(f"\n=== {title} ===", flush=True)
    print("[wall-clock]", flush=True)
    if metrics.prefill_wall_ms is None:
        print(
            f"  prefill : {'n/a':>12}       "
            f"(synthetic --fast prefill, no real throughput)",
            flush=True,
        )
    else:
        print(
            f"  prefill : {metrics.prefill_wall_ms:>12.3f} ms  "
            f"{metrics.prefill_tokens:>7d} tok  "
            f"{metrics.prefill_tok_s:>11.2f} tok/s",
            flush=True,
        )
        if metrics.cached_prefix_len > 0:
            print(
                f"          prefix-cache hit {metrics.cache_hit_frac * 100.0:>6.2f}%  "
                f"({metrics.cached_prefix_len}/{metrics.prompt_len} tok cached, "
                f"prefilled {metrics.prompt_len - metrics.cached_prefix_len} new)",
                flush=True,
            )
    print(
        f"  decode  : {d.wall_ms:>12.3f} ms  "
        f"{d.tokens:>7d} tok  "
        f"{d.tok_s:>11.2f} tok/s  "
        f"(batch={d.batch_size}, steps={d.steps}, ragged)",
        flush=True,
    )
    print("[kernel-time]  (decode_launch_wall; stream synchronization included)", flush=True)
    print(
        f"  decode  : {d.kernel_ms_total:>12.3f} ms  "
        f"mean/step {d.kernel_ms_mean:>10.4f} ms",
        flush=True,
    )


def _print_result_summary(
    result: dict[str, Any],
    timing_records: Sequence[tuple[str, float]] | None,
    *,
    title: str,
) -> None:
    """Emit the two timing reports plus a slim scalar-metrics line.

    (1) wall-clock + kernel summary (_print_run_report), (2) the per-step
    decode-launch timing table (_print_timing_table), and a one-line dump of any
    remaining scalar result fields with the bulky output/objects filtered out.
    """
    run_metrics = result.get("run_metrics")
    if isinstance(run_metrics, RunMetrics):
        _print_run_report(run_metrics, title=f"{title} -- throughput")
    scalar = {k: v for k, v in result.items() if k not in _NON_SCALAR_RESULT_KEYS}
    print(f"[mk-release] metrics = {scalar}", flush=True)
    _print_timing_table(list(timing_records or []), f"{title} (per-step decode ms)")


def _finalize_result_metrics(
    result: dict[str, Any],
    *,
    prefill_wall_ms: float | None,
    prompt_len: int,
    batch: int,
    cached_prefix_len: int,
) -> dict[str, Any]:
    """Wrap a decode result's DecodeMetrics into a full RunMetrics.

    The decode result carries ``decode_metrics`` (decode-only). This combines it
    with the prefill wall clock (None for synthetic ``--fast`` prefill) into
    ``result["run_metrics"]`` -- the single object callers print via
    ``_print_run_report``.
    """
    decode_metrics = result["decode_metrics"]
    run_metrics = RunMetrics(
        prompt_len=int(prompt_len),
        batch_size=int(batch),
        decode=decode_metrics,
        prefill_wall_ms=prefill_wall_ms,
        cached_prefix_len=int(cached_prefix_len),
    )
    result["run_metrics"] = run_metrics
    if prefill_wall_ms is not None:
        result["prefill_ms"] = float(prefill_wall_ms)
        result["prefill_tok_s"] = float(run_metrics.prefill_tok_s)
    return result


def _parse_true_false(value: str) -> bool:
    """Parse an argparse ``{true,false}`` value."""
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _parse_drain_sms_pair(value: str, *, flag: str, first: str, second: str) -> tuple[int, int]:
    """Parse an ``A,B`` drain-claimer pair (``--attn-drain-sms``/``--moe-drain-sms``)."""
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise SystemExit(f"{flag} must be 'A,B' ({first} count, {second} count)")
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        raise SystemExit(f"{flag} values must be integers, e.g. '64,132'") from None
    if a < 1 or b < 1:
        raise SystemExit(f"{flag} values must be >= 1")
    return a, b


def sample_ragged_cache_seqlens(
    *, batch_size: int, base_len: int, variance: int, seed: int,
) -> list[int]:
    """Sample absolute-uniform synthetic starting cache lengths.

    Draws each row's starting length uniformly from
    ``[base_len - variance, base_len + variance]`` (clamped at 1), which is the
    ragged fixture ``--ragged-batch`` benchmarks against.
    """
    if batch_size <= 0:
        return []
    base = max(1, int(base_len))
    spread = int(variance)
    if spread < 0:
        raise ValueError("seqlen variance must be non-negative")
    if spread == 0:
        return [base] * int(batch_size)
    lo = max(1, base - spread)
    hi = max(1, base + spread)
    # torch (already a hard dependency) rather than numpy, to keep the release
    # bundle's import surface small. The seed fixes this generator's stream, so
    # a given seed is reproducible within this runner.
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(
        lo, hi + 1, (int(batch_size),), generator=generator, dtype=torch.int32,
    ).tolist()


def resolve_fake_prompt_len_arg(
    fake_prompt_len_arg: Sequence[int],
    *,
    batch_size: int,
    ragged_batch: bool,
    seqlen_variance: int,
) -> tuple[int, list[int] | None, list[int] | None]:
    """Normalize ``--fake-prompt-len`` into (scalar base, seqlens, row_active).

    A single value returns ``(value, None, None)`` (uniform, all active).

    A list requires ``--ragged-batch`` and one entry per row. Each entry is either
    a positive starting cache length, or ``-1`` to mark that row as *masked*
    (inactive empty slot: ``cache_seqlen=0``, ``row_active=0``). Returns
    ``(max(active lengths), seqlens, row_active)``. Masking is how continuous-
    batch empty slots show up in the decode kernel; use it to reproduce mixed
    active/masked shapes from a server dump.
    """
    raw = [int(v) for v in fake_prompt_len_arg]
    if not raw:
        raise SystemExit("--fake-prompt-len requires at least one integer")
    if len(raw) == 1:
        if raw[0] == -1:
            raise SystemExit(
                "--fake-prompt-len -1 alone is invalid; masking requires a "
                "per-row list with --ragged-batch (use -1 for masked slots)")
        if raw[0] < 1:
            raise SystemExit(
                f"--fake-prompt-len must be >= 1 (or -1 per masked row with "
                f"--ragged-batch); got {raw[0]}")
        return raw[0], None, None
    if not ragged_batch:
        raise SystemExit(
            "--fake-prompt-len accepts multiple integers only with --ragged-batch")
    if int(seqlen_variance) != 0:
        raise SystemExit("when --fake-prompt-len is a list, --seqlen-variance must be 0")
    if len(raw) != int(batch_size):
        raise SystemExit(
            f"--fake-prompt-len list length ({len(raw)}) must match "
            f"--batch-size ({int(batch_size)})")
    seqlens: list[int] = []
    row_active: list[int] = []
    for v in raw:
        if v == -1:
            # Masked empty slot: matches continuous-batch inactive rows
            # (cache_seqlen=0, row_active=0). NOT the same as an active row at
            # length 0 — those are rejected so the CLI stays unambiguous.
            seqlens.append(0)
            row_active.append(0)
        elif v < 1:
            raise SystemExit(
                f"--fake-prompt-len entries must be >= 1 or -1 (masked); got {v}")
        else:
            seqlens.append(v)
            row_active.append(1)
    if not any(row_active):
        raise SystemExit(
            "--fake-prompt-len list cannot mask every row; need at least one "
            "active (>= 1) length")
    return max(seqlens), seqlens, row_active


def format_seqlen_summary(
    seqlens: Sequence[int],
    row_active: Sequence[int] | None,
) -> str:
    values = [int(v) for v in seqlens]
    if not values:
        return "empty"
    if row_active is None:
        active_bits = [1] * len(values)
    else:
        if len(row_active) != len(values):
            raise ValueError("row_active length must match seqlens")
        active_bits = [1 if int(a) != 0 else 0 for a in row_active]
    # Print masked slots as -1 so the CLI summary matches the input convention.
    display = [
        -1 if active_bits[i] == 0 else values[i] for i in range(len(values))
    ]
    active_lens = [values[i] for i in range(len(values)) if active_bits[i] != 0]
    n_masked = len(values) - len(active_lens)
    if not active_lens:
        return f"masked={n_masked} values={display}"
    return (
        f"min={min(active_lens)} mean={sum(active_lens) / len(active_lens):.1f} "
        f"max={max(active_lens)} masked={n_masked} values={display}"
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mk", action="store_true", help="select the release-native MK backend")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="native MK decode with a fake prompt (--fake-prompt-len): synthetic "
             "weights (unless --real-weight) and a noise-filled paged KV cache "
             "(no prefill forward).",
    )
    parser.add_argument(
        "--fast-prefill",
        dest="fast_prefill",
        action="store_true",
        help="with --mk --fast: run a REAL prefill forward over random fake "
             "prompts (synthetic weights) instead of noise-filling the KV "
             "cache. Use this to benchmark prefill; plain --fast stays cheap.",
    )
    parser.add_argument(
        "--real-weight",
        dest="real_weight",
        action="store_true",
        help="with --mk --fast: load real NMC checkpoint weights (--checkpoint) "
             "but keep a synthetic KV cache (no prefill). Incompatible with "
             "--fast-prefill and the non-fast E2E path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="local North Mini Code checkpoint directory. Required for real "
             "prompts and --real-weight. Synthetic --fast (no --real-weight) "
             "uses the baked-in NMC geometry when this is omitted.",
    )
    parser.add_argument(
        "--lib-path",
        default=None,
        help="absolute path to libmk_release.so",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="single user prompt (broadcast to --batch-size rows)")
    parser.add_argument(
        "--msgs", type=str, nargs="+", default=None,
        help="one prompt per decode row (ragged real batch); overrides "
             "--prompt and sets --batch-size to the number of prompts. "
             "Real-prompt E2E path only (not --fast).")
    parser.add_argument("--batch-size", "--bs", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="MK decode: generated buffer width including the prefill bootstrap "
             "token (column 0). Decode runs (max_new_tokens - 1) steps and "
             "keeps the bootstrap in completion text/IDs while excluding it "
             "from decode-loop throughput counters. "
             "Non-MK NmcRunner still treats this as the number of sampled tokens.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="nucleus sampling cutoff for the real-weight prefill sampler")
    parser.add_argument("--gumbel-sampler", action="store_true",
                        help="when --temperature is nonzero, sample the MK decode with "
                             "Gumbel-max instead of torch.multinomial (Python decode loop only)")
    parser.add_argument("--cpp-decode-runtime", "--cpp-decode", dest="cpp_decode_runtime",
                        action="store_true",
                        help="run the whole decode loop in C++ (launch.runtime_generate): "
                             "KV step, scratch/barrier reset, layer-0 seed, NMC launch and "
                             "sampling all happen in C++. Greedy at temperature 0, native "
                             "Gumbel-max otherwise; requires --top-p 1 and ignores "
                             "--gumbel-sampler/--decode-timeout-sec")
    parser.add_argument("--fake-prompt-len", type=int, nargs="+",
                        default=[DEFAULT_FAKE_PROMPT_LENGTH],
                        help="starting KV-cache length for --fast synthetic decode. "
                             "Accepts one base length, or (with --ragged-batch) one "
                             "explicit length per --batch-size row. Use -1 for a "
                             "masked (inactive empty) slot in a per-row list.")
    parser.add_argument(
        "--rr-dependency-affinity",
        action="store_true",
        help="experimental RR A/B: keep wave order and per-SM instruction "
             "counts, but co-locate static attention/combine and "
             "router/top-k/finalize/gather dependency chains",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="insert a GRID_SYNC on every SM between dependency waves so "
             "wave N+1 cannot overlap unfinished wave-N producers. No extra "
             "scratch buffers; the schedule instruction list grows only. "
             "With --profile, GRID_SYNC ranges appear in the Perfetto trace.",
    )
    parser.add_argument("--attn-drain", dest="attn_drain",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="use ATTN_DRAIN on FULL-attention layers for "
                             "load-balanced (ragged) attention; sliding layers "
                             "keep the static wave. Default ON; pass "
                             "--no-attn-drain for the static A/B reference arm. "
                             "Supported by both the Python step loop and "
                             "--cpp-decode-runtime.")
    parser.add_argument("--attn-drain-sms", dest="attn_drain_sms", type=int, default=None,
                        metavar="N",
                        help="number of ATTN_DRAIN claimer instructions per "
                             "FULL-attention layer. Sliding layers never drain. "
                             "Any value >=1 is correct; this only tunes claim "
                             "parallelism. Requires attn drain; defaults to "
                             "--num-sms (pin to 132 on H100).")
    parser.add_argument("--moe-drain-sms", dest="moe_drain_sms", type=str, default=None,
                        metavar="A,B",
                        help="number of MoE drain claimer instructions per MoE layer: "
                             "A for the upgate-act drain, B for the down drain "
                             "(format 'A,B'). The release always runs the dynamic MoE "
                             "scheduler, where each claimer atomically pulls expert "
                             "tiles from the shared queue until drained, so any value "
                             ">=1 is correct. Defaults to --num-sms for both.")
    parser.add_argument(
        "--moe-combine-atomic-tma",
        dest="moe_combine_atomic_tma",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="A/B arm: remove MOE_COMBINE and have MoE down scatter-reduce "
             "directly into x_ffn via TMA reduce-add (bf16). "
             "Default OFF (baseline combine op).",
    )
    parser.add_argument("--ragged-batch", dest="ragged_batch", action="store_true",
                        help="synthetic --fast mode: sample per-row starting cache "
                             "lengths around --fake-prompt-len (see --seqlen-variance).")
    parser.add_argument("--seqlen-variance", dest="seqlen_variance", type=int, default=0,
                        help="absolute uniform half-width around --fake-prompt-len "
                             "for --ragged-batch")
    parser.add_argument("--ragged-seed", dest="ragged_seed", type=int, default=1,
                        help="random seed for synthetic ragged sequence lengths")
    # Defaults are resolved in _run_mk from decode_schedule's constants rather than
    # duplicated here, so there is one source of truth for the split policy.
    parser.add_argument("--max-attn-splits", dest="max_attn_splits", type=int, default=None,
                        help="cap on split-KV parallelism per attention task "
                             "(default: decode_schedule.DEFAULT_MAX_ATTN_SPLITS)")
    parser.add_argument("--min-attn-chunk", dest="min_attn_chunk", type=int, default=None,
                        help="floor on tokens covered by one attention split; must be "
                             "a positive multiple of --page-block "
                             "(default: decode_schedule.DEFAULT_MIN_ATTN_CHUNK)")
    parser.add_argument("--num-sms", type=int, default=DEFAULT_NUM_SMS)
    parser.add_argument("--page-block", type=int, default=DEFAULT_PAGE_BLOCK)
    parser.add_argument("--prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE)
    parser.add_argument("--prefill-warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS,
                        help="--mk --fast --fast-prefill only: discarded prefill passes "
                             "before timing (excludes one-time autotune)")
    parser.add_argument("--prefill-iters", type=int, default=DEFAULT_TIMED_ITERATIONS,
                        help="--mk --fast --fast-prefill only: timed prefill passes; "
                             "the median is reported")
    parser.add_argument("--frac-vram-utilization", type=float, default=DEFAULT_VRAM_UTILIZATION)
    parser.add_argument("--decode-timeout-sec", type=float, default=1.0,
                        help="per-step decode wall-time budget; exits with code 124 when a "
                             "step overruns (<= 0 disables the check)")
    parser.add_argument(
        "--profile",
        default=None,
        metavar="PATH",
        help="--mk Python-loop only: write a compact per-block SM-profiler "
             "Perfetto trace for the first decode step. The normal and profiled "
             "kernels are warmed first, and a same-state non-profile kernel "
             "measurement is printed for comparison.",
    )
    parser.add_argument(
        "--profile-max-events",
        type=int,
        default=DEFAULT_PROFILE_MAX_EVENTS,
        metavar="N",
        help="maximum compact profiler range segments per block "
             f"(default: {DEFAULT_PROFILE_MAX_EVENTS})",
    )
    parser.add_argument("--print-first", type=int, default=-1,
                        help="print the first N batch completions (-1 = all)")
    parser.add_argument("--print-ids", type=_parse_true_false, default=True, metavar="{true,false}",
                        help="print generated token ids in the final summary (default: true); "
                             "always suppressed under --fast where ids are synthetic")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="host-side hang diagnostics: write device pointer "
                             "maps under ./dump for cuda-gdb, and let the C++ "
                             "decode-service watchdog take a device memory "
                             "snapshot and park on the hung stream instead of "
                             "exiting")
    return parser.parse_args(list(argv))


def _release_library_path(value: str | None) -> str:
    """Resolve the explicit or conventional absolute release-library path."""
    if value is not None:
        return os.path.abspath(value)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, "build", "libmk_release.so")


def _run_native_benchmark(
    args: argparse.Namespace, params: SamplingParams
) -> NativeRunReport:
    """Run the real release MK prefill plus RR/all megakernel decode.

    The decode launches one broadcast batch; the report keeps per-row token ids
    so the printer can show individual completions without ever launching
    `batch_size` independent singleton requests.
    """
    try:
        import decode.schedule as decode_schedule
        import native
        import prefill.torch_prefill as torch_prefill
    except ImportError as exc:
        raise RuntimeError(
            "native runner requires sibling prefill/torch_prefill.py and decode/schedule.py"
        ) from exc

    # The only place --lib and --debug are consumed. Everything downstream
    # reaches the ABI through native.ext() and never sees a path.
    native.bind(
        library_path=_release_library_path(args.lib_path),
        debug=bool(args.debug),
    )
    device = torch.device(str(args.device))
    dtype = _dtype_from_name(str(args.dtype))
    if dtype != torch.bfloat16:
        raise ValueError(
            "--mk uses the BF16-only native decode kernel; pass --dtype bf16"
        )

    # Ragged multi-prompt: one real prompt string per decode row. Only the
    # non--fast E2E path consumes per-row prompt strings; --fast builds a
    # synthetic KV cache and has its own --ragged-batch knob instead.
    prompts_list: list[str] | None = None
    if args.msgs:
        if bool(args.fast):
            raise SystemExit(
                "--msgs (ragged real prompts) is not supported with --fast; use "
                "--ragged-batch for the synthetic --fast KV cache instead")
        if bool(args.real_weight):
            raise SystemExit(
                "--msgs is the real-prompt E2E path; do not combine with "
                "--real-weight (that keeps a synthetic KV cache)")
        prompts_list = [str(m) for m in args.msgs]
        if int(args.batch_size) != len(prompts_list):
            print(
                f"[mk-release] --msgs given: overriding --batch-size "
                f"{args.batch_size} -> {len(prompts_list)} (one row per prompt)",
                flush=True,
            )
        args.batch_size = len(prompts_list)

    batch_size = int(args.batch_size)
    if batch_size not in decode_schedule.SUPPORTED_BATCH_SIZES:
        raise ValueError(
            f"--batch-size must be one of {decode_schedule.SUPPORTED_BATCH_SIZES} for --mk"
        )
    if int(args.num_sms) < 1:
        raise ValueError("--num-sms must be positive")
    if int(args.page_block) < 1:
        raise ValueError("--page-block must be positive")
    if int(args.prefill_chunk_size) < 1:
        raise ValueError("--prefill-chunk-size must be positive")
    cpp_runtime = bool(args.cpp_decode_runtime)
    if cpp_runtime and float(args.top_p) != 1.0:
        raise ValueError("--cpp-decode-runtime supports only --top-p 1.0")
    if args.profile is not None:
        if cpp_runtime:
            raise ValueError("--profile cannot be combined with --cpp-decode-runtime")
        if int(args.profile_max_events) <= 0:
            raise ValueError("--profile-max-events must be positive")
        if int(params.max_new_tokens) < 2:
            raise ValueError("--profile requires --max-new-tokens >= 2")
    if bool(args.fast_prefill):
        if not bool(args.fast):
            raise ValueError("--fast-prefill requires --fast")
        if not bool(args.mk):
            raise ValueError("--fast-prefill requires --mk")
        if bool(args.real_weight):
            raise ValueError("--fast-prefill is incompatible with --real-weight")
    if bool(args.real_weight):
        if not bool(args.fast):
            raise ValueError("--real-weight requires --fast")
        if not bool(args.mk):
            raise ValueError("--real-weight requires --mk")

    max_attn_splits = (
        decode_schedule.DEFAULT_MAX_ATTN_SPLITS if args.max_attn_splits is None
        else int(args.max_attn_splits))
    min_attn_chunk = (
        decode_schedule.DEFAULT_MIN_ATTN_CHUNK if args.min_attn_chunk is None
        else int(args.min_attn_chunk))
    if max_attn_splits < 1:
        raise ValueError("--max-attn-splits must be positive")
    if min_attn_chunk < 1 or min_attn_chunk % int(args.page_block) != 0:
        raise ValueError("--min-attn-chunk must be a positive multiple of --page-block")

    attn_drain = bool(args.attn_drain)
    attn_drain_sms: int | None = None
    if args.attn_drain_sms is not None:
        if not attn_drain:
            raise SystemExit("--attn-drain-sms requires attn drain "
                             "(omit --no-attn-drain)")
        attn_drain_sms = int(args.attn_drain_sms)
        if attn_drain_sms < 1:
            raise SystemExit("--attn-drain-sms must be >= 1")
    moe_drain_sms: tuple[int, int] | None = None
    if args.moe_drain_sms is not None:
        # Independent of --attn-drain: the MoE drains exist on every MoE layer
        # regardless of how attention is scheduled.
        moe_drain_sms = _parse_drain_sms_pair(
            args.moe_drain_sms, flag="--moe-drain-sms", first="upgate", second="down")
    # Resolve --fake-prompt-len into a scalar base plus (for ragged fixtures) the
    # explicit per-row cache lengths / optional row_active mask (-1 → masked).
    fake_prompt_len, explicit_cache_seqlens, explicit_row_active = (
        resolve_fake_prompt_len_arg(
            args.fake_prompt_len,
            batch_size=batch_size,
            ragged_batch=bool(args.ragged_batch),
            seqlen_variance=int(args.seqlen_variance),
        )
    )
    if bool(args.fast_prefill) and explicit_cache_seqlens is not None:
        raise ValueError(
            "--fast-prefill cannot produce ragged cache lengths; omit "
            "per-row --fake-prompt-len / --ragged-batch"
        )
    if bool(args.ragged_batch):
        if not bool(args.fast):
            raise SystemExit("--ragged-batch is currently supported only with --fast")
        if bool(args.fast_prefill):
            raise SystemExit("--ragged-batch is incompatible with --fast-prefill")
        if explicit_cache_seqlens is None:
            explicit_cache_seqlens = sample_ragged_cache_seqlens(
                batch_size=batch_size,
                base_len=fake_prompt_len,
                variance=int(args.seqlen_variance),
                seed=int(args.ragged_seed),
            )
            explicit_row_active = None  # sampled rows are all active
        print(
            f"[mk-release] ragged batch initial_cache_seqlens "
            f"{format_seqlen_summary(explicit_cache_seqlens, explicit_row_active)}",
            flush=True,
        )
    else:
        if int(args.seqlen_variance) != 0:
            raise SystemExit("--seqlen-variance requires --ragged-batch")
        explicit_cache_seqlens = None
        explicit_row_active = None

    needs_checkpoint = (not bool(args.fast)) or bool(args.real_weight)
    if needs_checkpoint and not args.checkpoint:
        raise SystemExit(
            "--checkpoint is required unless --fast is set without --real-weight"
        )
    if args.checkpoint:
        print(
            f"[mk-release] loading NMC config checkpoint={args.checkpoint} "
            f"device={device} dtype={dtype}",
            flush=True,
        )
        cfg = torch_prefill.NmcConfig.from_checkpoint(str(args.checkpoint))
    else:
        print(
            f"[mk-release] using baked-in NMC release geometry "
            f"device={device} dtype={dtype}",
            flush=True,
        )
        cfg = torch_prefill.NmcConfig.release()
    prefill = None
    try:
        if bool(args.fast):
            # Plain --fast: noise-fill
            # the paged KV and skip the prefill forward. Opt into a real forward
            # with --fast-prefill (prefill microbench only).
            if bool(args.fast_prefill):
                print(
                    f"[mk-release] FAST-PREFILL: starting real prefill forward "
                    f"batch={batch_size} prompt_len={fake_prompt_len} "
                    f"max_new={params.max_new_tokens}",
                    flush=True,
                )
                fast = torch_prefill.run_fast_prefill(
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                    batch=batch_size,
                    fake_prompt_len=fake_prompt_len,
                    max_new_tokens=params.max_new_tokens,
                    page_block=int(args.page_block),
                    frac_vram_utilization=float(args.frac_vram_utilization),
                    prefill_chunk_size=int(args.prefill_chunk_size),
                    upgate_chunk=DEFAULT_UPGATE_LAYOUT_CHUNK,
                    moe_upgate_chunk=DEFAULT_MOE_UPGATE_LAYOUT_CHUNK,
                    warmup_iterations=int(args.prefill_warmup),
                    timed_iterations=int(args.prefill_iters),
                )
                prefill = fast["prefill"]
                weights = fast["weights"]
                prefill_ms = float(fast["prefill_ms"])
                tokenizer = None
                print(
                    f"[mk-release] FAST-PREFILL: prefill ready time={prefill_ms:.3f} ms "
                    f"cache_seqlen={prefill.prompt_len}",
                    flush=True,
                )
            else:
                fake_len_arg: int | list[int] = (
                    explicit_cache_seqlens if explicit_cache_seqlens is not None
                    else fake_prompt_len)
                weight_kind = "REAL-WEIGHT" if args.real_weight else "SYNTH-WEIGHT"
                print(
                    f"[mk-release] FAST+{weight_kind}: loading weights + "
                    f"synthetic KV batch={batch_size} "
                    f"prompt_len={fake_len_arg} max_new={params.max_new_tokens}",
                    flush=True,
                )
                started = time.monotonic_ns()
                if bool(args.real_weight):
                    weights = torch_prefill.load_weights(
                        checkpoint=str(args.checkpoint),
                        cfg=cfg,
                        device=device,
                        dtype=dtype,
                        upgate_chunk=DEFAULT_UPGATE_LAYOUT_CHUNK,
                        moe_upgate_chunk=DEFAULT_MOE_UPGATE_LAYOUT_CHUNK,
                    )
                else:
                    weights = torch_prefill.make_fast_nmc_weights(
                        cfg=cfg,
                        device=device,
                        dtype=dtype,
                        upgate_chunk=DEFAULT_UPGATE_LAYOUT_CHUNK,
                        moe_upgate_chunk=DEFAULT_MOE_UPGATE_LAYOUT_CHUNK,
                    )
                torch.cuda.synchronize(device)
                weights_ms = (
                    time.monotonic_ns() - started
                ) / (NANOSECONDS_PER_SECOND / MILLISECONDS_PER_SECOND)
                started = time.monotonic_ns()
                prefill = torch_prefill.make_synthetic_kv_prefill(
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                    batch=batch_size,
                    fake_prompt_len=fake_len_arg,
                    max_new_tokens=params.max_new_tokens,
                    page_block=int(args.page_block),
                    frac_vram_utilization=float(args.frac_vram_utilization),
                    row_active=explicit_row_active,
                )
                torch.cuda.synchronize(device)
                kv_ms = (
                    time.monotonic_ns() - started
                ) / (NANOSECONDS_PER_SECOND / MILLISECONDS_PER_SECOND)
                prefill_ms = None  # synthetic KV: no real prefill throughput
                tokenizer: Tokenizer | None = None
                print(
                    f"[mk-release] FAST+{weight_kind}: weights={weights_ms:.3f} ms "
                    f"kv_setup={kv_ms:.3f} ms max_cache_seqlen={prefill.prompt_len}",
                    flush=True,
                )
        else:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("real native mode requires transformers") from exc
            print("[mk-release] E2E: loading tokenizer", flush=True)
            tokenizer_impl = AutoTokenizer.from_pretrained(
                str(args.checkpoint),
                trust_remote_code=False,
                local_files_only=True,
            )
            tokenizer = TransformersTokenizer(tokenizer_impl)
            print("[mk-release] E2E: loading MK-layout weights", flush=True)
            weights = torch_prefill.load_weights(
                checkpoint=str(args.checkpoint),
                cfg=cfg,
                device=device,
                dtype=dtype,
                upgate_chunk=DEFAULT_UPGATE_LAYOUT_CHUNK,
                moe_upgate_chunk=DEFAULT_MOE_UPGATE_LAYOUT_CHUNK,
            )
            if prompts_list is not None:
                print(
                    f"[mk-release] E2E: starting ragged multi-prompt prefill "
                    f"batch={batch_size} max_new={params.max_new_tokens}",
                    flush=True,
                )
            else:
                print(
                    f"[mk-release] E2E: starting prefill batch={batch_size} "
                    f"max_new={params.max_new_tokens}",
                    flush=True,
                )
            started = time.monotonic_ns()
            prefill = torch_prefill.run_reference_prefill(
                prompt=str(args.prompt),
                batch=batch_size,
                weights=weights,
                tokenizer=tokenizer_impl,
                cfg=cfg,
                device=device,
                dtype=dtype,
                raw_prompt=bool(args.raw_prompt),
                page_block=int(args.page_block),
                prefill_chunk_size=int(args.prefill_chunk_size),
                frac_vram_utilization=float(args.frac_vram_utilization),
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=float(args.top_p),
                prompts=prompts_list,
            )
            torch.cuda.synchronize(device)
            prefill_ms = (
                time.monotonic_ns() - started
            ) / (NANOSECONDS_PER_SECOND / MILLISECONDS_PER_SECOND)
            row_seqlens = [
                int(v) for v in
                prefill.kv_cache.cache_seqlens[:batch_size].detach().cpu().tolist()
            ]
            if prompts_list is not None:
                print(
                    f"[mk-release] E2E: prefill ready time={prefill_ms:.3f} ms "
                    f"prompt_len={prefill.prompt_len} "
                    f"cache_seqlens={format_seqlen_summary(row_seqlens, None)}",
                    flush=True,
                )
            else:
                print(
                    f"[mk-release] E2E: prefill ready time={prefill_ms:.3f} ms "
                    f"prompt_len={prefill.prompt_len}",
                    flush=True,
                )

        if prefill_ms is None:
            print(
                "[mk-release] prefill throughput=n/a "
                "(synthetic KV / --real-weight, no real prefill)",
                flush=True,
            )
        else:
            # Uniform: batch * prompt_len. Ragged multi-prompt: sum of true
            # per-row lengths (pad tokens are not "prompt tokens served").
            if prompts_list is not None:
                prefill_tokens = sum(
                    int(v) for v in
                    prefill.kv_cache.cache_seqlens[:batch_size].detach().cpu().tolist()
                )
            else:
                prefill_tokens = batch_size * int(prefill.prompt_len)
            prefill_tok_s = prefill_tokens / max(prefill_ms / MILLISECONDS_PER_SECOND, 1e-9)
            print(
                f"[mk-release] prefill throughput={prefill_tok_s:.2f} tok/s "
                f"tokens={prefill_tokens} time={prefill_ms:.3f} ms",
                flush=True,
            )

        decode = decode_schedule.run_mk_decode_from_prefill(
            prefill=prefill,
            weights=weights,
            cfg=cfg,
            num_sms=int(args.num_sms),
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            gumbel_sampler=bool(args.gumbel_sampler),
            max_attn_splits=max_attn_splits,
            min_attn_chunk=min_attn_chunk,
            fast=bool(args.fast),
            decode_timeout_sec=float(args.decode_timeout_sec),
            cpp_runtime=cpp_runtime,
            attn_drain=attn_drain,
            attn_drain_sms=attn_drain_sms,
            moe_drain_sms=moe_drain_sms,
            profile_path=None if args.profile is None else str(args.profile),
            profile_max_events=int(args.profile_max_events),
            rr_dependency_affinity=bool(args.rr_dependency_affinity),
            moe_combine_atomic_tma=bool(args.moe_combine_atomic_tma),
            ablation=bool(args.ablation),
        )
        rows = tuple(
            tuple(int(token_id) for token_id in row)
            for row in decode["token_ids_by_batch"]
        )
        active = decode.get("row_active")
        if active is None:
            ragged_tokens = sum(len(row) for row in rows)
            metrics_batch = batch_size
        else:
            # Exclude masked rows from billed decode tokens / effective batch.
            ragged_tokens = sum(
                len(row) for i, row in enumerate(rows) if int(active[i]) != 0
            )
            metrics_batch = int(decode.get("num_active_rows", sum(1 for a in active if int(a) != 0)))
        decode["decode_metrics"] = DecodeMetrics(
            wall_ms=float(decode["decode_wall_ms"]),
            kernel_ms_total=sum(float(value) for value in decode["decode_step_ms"]),
            steps=len(decode["decode_step_ms"]),
            tokens=ragged_tokens,
            batch_size=metrics_batch,
        )
        # Real prefill has a meaningful throughput; synthetic --fast prefill /
        # --real-weight synthetic KV do not, so wall clock is dropped to None.
        result = _finalize_result_metrics(
            decode,
            prefill_wall_ms=None if bool(args.fast) else prefill_ms,
            prompt_len=int(decode["prompt_len"]),
            batch=batch_size,
            cached_prefix_len=0,
        )
        timing_records = [("decode_launch_wall", float(ms)) for ms in decode["decode_step_ms"]]
        texts = None if tokenizer is None else tuple(tokenizer.decode(row) for row in rows)
        if bool(args.fast) and bool(args.fast_prefill):
            mode_tag = "fast-prefill"
        elif bool(args.fast) and bool(args.real_weight):
            mode_tag = "fast+real-weight"
        elif bool(args.fast):
            mode_tag = "fast"
        else:
            mode_tag = "real"
        return NativeRunReport(
            result=result,
            timing_records=timing_records,
            title=(
                f"NMC MK Release mode=all "
                f"({mode_tag}, "
                f"{'cpp-runtime' if cpp_runtime else 'python-loop'})"
            ),
            texts=texts,
            token_ids_by_batch=rows,
            fast=bool(args.fast),
            batch_size=batch_size,
        )
    finally:
        if prefill is not None:
            prefill.close()


def _print_native_report(report: NativeRunReport, *, print_first: int, print_ids: bool) -> None:
    """Print the throughput summary plus per-row completions.

    ``--fast`` suppresses text and ids entirely: synthetic logits do not decode
    to anything meaningful, so echoing them would only mislead.
    """
    _print_result_summary(report.result, report.timing_records, title=report.title)
    if report.fast:
        print("[mk-release] synthetic output suppressed (--fast)", flush=True)
        return
    count = len(report.token_ids_by_batch)
    shown = count if print_first < 0 else min(count, print_first)
    for row_index in range(shown):
        if report.texts is not None:
            print(f"[mk-release] completion[{row_index}]: {report.texts[row_index]!r}", flush=True)
        if print_ids:
            print(
                f"[mk-release] token_ids[{row_index}]: "
                f"{list(report.token_ids_by_batch[row_index])}",
                flush=True,
            )


def main(argv: Sequence[str]) -> None:
    """Run one broadcast benchmark and print prefill/decode throughput."""
    args = _parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.msgs and not bool(args.mk):
        raise SystemExit("--msgs requires --mk (ragged real multi-prompt E2E)")
    params = SamplingParams(
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
    )
    params.validate()
    if args.profile is not None and not bool(args.mk):
        raise SystemExit("--profile requires --mk")
    if bool(args.mk):
        report = _run_native_benchmark(args, params)
        _print_native_report(
            report, print_first=int(args.print_first), print_ids=bool(args.print_ids))
        return
    runner = NmcRunner(
        checkpoint=str(args.checkpoint),
        device_name=str(args.device),
        dtype_name=str(args.dtype),
        fast=bool(args.fast),
    )
    try:
        # Benchmark-only broadcast: the server never duplicates user prompts this way.
        results = [runner.generate(str(args.prompt), params) for _ in range(int(args.batch_size))]
        prompt_len = len(runner.tokenizer.encode(str(args.prompt)))
        prefill_seconds = max(result.prefill_seconds for result in results)
        decode_seconds = sum(result.decode_seconds for result in results)
        generated_tokens = sum(len(result.token_ids) for result in results)
        steps = max((len(result.token_ids) for result in results), default=0)
        # The reference path has no separate megakernel, so kernel time equals
        # the PyTorch decode wall time (the forward passes ARE the compute).
        decode_metrics = DecodeMetrics(
            wall_ms=decode_seconds * MILLISECONDS_PER_SECOND,
            kernel_ms_total=decode_seconds * MILLISECONDS_PER_SECOND,
            steps=steps,
            tokens=generated_tokens,
            batch_size=int(args.batch_size),
        )
        run_metrics = RunMetrics(
            prompt_len=prompt_len,
            batch_size=int(args.batch_size),
            decode=decode_metrics,
            prefill_wall_ms=prefill_seconds * MILLISECONDS_PER_SECOND,
            cached_prefix_len=0,
        )
        _print_run_report(run_metrics, title="NMC PyTorch reference mode=all -- throughput")
        count = len(results)
        shown = count if int(args.print_first) < 0 else min(count, int(args.print_first))
        for row_index in range(shown):
            print(
                f"[mk-release] completion[{row_index}]: {results[row_index].text!r}",
                flush=True,
            )
            if bool(args.print_ids):
                print(
                    f"[mk-release] token_ids[{row_index}]: "
                    f"{list(results[row_index].token_ids)}",
                    flush=True,
                )
    finally:
        runner.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])