"""NMC-only OpenAI-compatible server backed by the release megakernel."""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

# Package imports below resolve against ``src/`` when launched as
# ``python src/serving/server.py``, where sys.path[0] is ``src/serving``.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fastapi import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JSONResponse,
    StreamingResponse,
)
from serving.parse import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    Cmd4StreamParser,
    MelodyParseError,
    ParsedAssistantDelta,
    ParsedToolCall,
    parse_cmd4,
    require_cohere_melody,
)

DEFAULT_NMC_MODEL_ID = "North-Mini-Code-1.0"
DEFAULT_MODEL_ID = DEFAULT_NMC_MODEL_ID
_ROW_OUTPUT_RE = re.compile(r"^ids=(?P<ids>\[.*\]) :: (?P<text>.*)$", re.DOTALL)
SUPPORTED_BATCH_SIZES = (1, 2, 4, 8)
LOG = logging.getLogger(__name__)

# Sampling knobs this server actually honors. Known unsupported OpenAI knobs
# are rejected in _params_from_payload rather than silently changing request
# semantics; unknown extension fields remain ignored for client compatibility.
_SUPPORTED_SAMPLING_SUMMARY = (
    "supported sampling params: max_tokens / max_completion_tokens (>0), "
    "temperature (>=0; 0=greedy), "
    "top_p (must be 1.0), top_k (1=greedy or unset), stop (string or list of strings); "
    "unsupported: top_k>1, frequency_penalty, presence_penalty, logit_bias, n!=1, "
    "logprobs, seed (per-request)"
)


@dataclasses.dataclass(frozen=True)
class BatchParams:
    max_tokens: int
    temperature: float
    top_p: float = 1.0
    stop: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class GeneratedText:
    text: str
    raw_text: str | None = None
    token_ids: tuple[int, ...] = ()
    finish_reason: str = "length"
    prompt_token_count: int = 0


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    raw: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ToolChoice:
    mode: str
    function_name: str | None


class ToolCallGenerationError(RuntimeError):
    """Raised when model output violates the requested tool-call contract."""


@dataclasses.dataclass(frozen=True)
class ToolCallPolicy:
    mode: str
    function_name: str | None
    allowed_names: frozenset[str]


@dataclasses.dataclass
class MkServerConfig:
    model_id: str = DEFAULT_MODEL_ID
    bs: int = 4
    max_new: int = 128
    temperature: float = 0.0
    lib_path: str | None = None
    frac_vram_utilization: float = 0.90
    stream_chunk_size: int = 128
    nmc_ckpt: str | None = None
    nmc_dtype: str = "bf16"
    nmc_device: str = "cuda"
    nmc_page_block: int = 64
    nmc_prefill_chunk_size: int = 1024
    nmc_num_sms: int = 132
    nmc_max_attn_splits: int = 32
    nmc_min_attn_chunk: int = 512
    nmc_max_context: int = 262144
    nmc_batched_prefill: bool = True
    nmc_batched_prefill_max_batch: int = 8
    nmc_batched_prefill_token_budget: int = 8192
    nmc_batched_prefill_coalesce_ms: float = 0.5
    # Default ON: full-attention layers drain with a per-step host-built queue.
    # Pass --no-attn-drain for the static A/B reference arm.
    nmc_attn_drain: bool = True
    nmc_attn_drain_sms: int | None = None
    # Host-side debug diagnostics only (device pointer maps, watchdog device
    # snapshot, and a park on the hung stream for cuda-gdb instead of exiting).
    # Does not change decode behaviour or the compiled kernel.
    debug: bool = False

    def validate(self) -> None:
        if not self.nmc_ckpt:
            raise ValueError("--ckpt is required")
        if self.bs not in SUPPORTED_BATCH_SIZES:
            raise ValueError(f"bs must be one of {SUPPORTED_BATCH_SIZES}")
        if self.max_new < 1:
            raise ValueError("max_new must be positive")
        if self.nmc_max_context < 2:
            raise ValueError("nmc_max_context must be at least 2")
        if self.max_new >= self.nmc_max_context:
            raise ValueError("max_new must be smaller than nmc_max_context")
        if self.nmc_max_context > 2_147_483_647:
            raise ValueError("nmc_max_context must fit in the native int32 ABI")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if self.stream_chunk_size < 1:
            raise ValueError("stream_chunk_size must be positive")
        if not (0.0 < float(self.frac_vram_utilization) < 1.0):
            raise ValueError("frac_vram_utilization must be in (0, 1)")
        if self.nmc_page_block < 1:
            raise ValueError("nmc_page_block must be positive")
        if self.nmc_prefill_chunk_size < 1:
            raise ValueError("nmc_prefill_chunk_size must be positive")
        if str(self.nmc_dtype).lower() not in {"bf16", "bfloat16"}:
            raise ValueError(
                "nmc_dtype must be bf16/bfloat16; native decode is BF16-only"
            )
        if self.nmc_min_attn_chunk < 1 or self.nmc_min_attn_chunk % self.nmc_page_block:
            raise ValueError("nmc_min_attn_chunk must be a positive page-block multiple")
        if self.nmc_max_attn_splits < 1 or self.nmc_num_sms < 1:
            raise ValueError("NMC schedule and context settings must be positive")
        if self.nmc_batched_prefill_max_batch < 2:
            raise ValueError("nmc_batched_prefill_max_batch must be >= 2")
        if self.nmc_batched_prefill_token_budget < 2:
            raise ValueError("nmc_batched_prefill_token_budget must be >= 2")
        coalesce_ms = float(self.nmc_batched_prefill_coalesce_ms)
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0.0:
            raise ValueError("nmc_batched_prefill_coalesce_ms must be non-negative")
        if self.nmc_attn_drain_sms is not None:
            if not self.nmc_attn_drain:
                raise ValueError("nmc_attn_drain_sms requires nmc_attn_drain")
            if int(self.nmc_attn_drain_sms) < 1:
                raise ValueError("nmc_attn_drain_sms must be >= 1")
            if int(self.nmc_attn_drain_sms) > int(self.nmc_num_sms):
                raise ValueError("nmc_attn_drain_sms cannot exceed nmc_num_sms")


class NmcMkSessionEngine:
    """Continuous-batching NMC engine backed by ``NmcDecodeSession``.

    The engine owns a long-lived decode service and admits each request into a
    free slot, so requests can decode concurrently at the current session
    geometry. Requests are independent; there is no global GPU lock here (the
    session scheduler serializes pause-guarded admission and eviction).
    """

    def __init__(self, config: MkServerConfig) -> None:
        config.validate()
        self.config = config
        self._kv_reservation_tokens = int(config.max_new)
        import native
        import prefill.torch_prefill as torch_prefill
        import torch
        from transformers import AutoTokenizer

        # Bind the process to one build before anything else: this is the only
        # place --lib is consumed, and failing here costs nothing, whereas
        # discovering a missing build after the multi-minute weight load does.
        root = os.path.abspath(os.path.join(_SRC_DIR, ".."))
        native.bind(
            library_path=os.path.abspath(
                config.lib_path or os.path.join(root, "build", "libmk_release.so")
            ),
            debug=bool(config.debug),
        )

        if not config.nmc_ckpt:
            raise ValueError("--ckpt is required")
        checkpoint = config.nmc_ckpt
        self.device = torch.device(config.nmc_device)
        if self.device.type != "cuda":
            raise ValueError("NMC release server requires a CUDA device")
        self.dtype = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        }[str(config.nmc_dtype).lower()]
        LOG.info("loading NMC config from %s", checkpoint)
        self.cfg = torch_prefill.NmcConfig.from_checkpoint(checkpoint)
        LOG.info("loading NMC tokenizer from %s", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=False, local_files_only=True)
        self.runner = SimpleNamespace(tokenizer=tokenizer)
        LOG.info(
            "loading NMC weights checkpoint=%s device=%s dtype=%s",
            checkpoint,
            self.device,
            self.dtype,
        )
        self.weights = torch_prefill.load_weights(
            checkpoint=checkpoint, cfg=self.cfg, device=self.device, dtype=self.dtype,
            upgate_chunk=64, moe_upgate_chunk=64)
        LOG.info("loaded NMC weights checkpoint=%s", checkpoint)
        self._session: Any | None = None

    def warmup_prefill(self) -> None:
        if self._session is not None:
            return
        import prefill.torch_prefill as torch_prefill
        import torch
        from kv.pool import get_nmc_shared_kv_arena
        from serving.session import NmcDecodeSession
        started = time.perf_counter()

        # The session's B=1 row-prefill path is the serving prefill path. Build
        # both its normal configured chunk and its one-token tail before serving
        # requests, while sizing the shared arena for the largest decode geometry
        # that will be created immediately afterward.
        prompt_len = int(self.config.nmc_prefill_chunk_size) + 1
        decode_pages = (
            self._kv_reservation_tokens + int(self.config.nmc_page_block) - 1
        ) // int(self.config.nmc_page_block)
        warmup_pages = (
            prompt_len + int(self.config.nmc_page_block) - 1
        ) // int(self.config.nmc_page_block)
        get_nmc_shared_kv_arena(
            required_blocks=(
                int(self.cfg.num_hidden_layers)
                * max(
                    int(self.config.bs) * max(1, decode_pages),
                    max(1, warmup_pages),
                )
            ),
            page_block=int(self.config.nmc_page_block),
            n_kv_heads=int(self.cfg.num_key_value_heads),
            head_dim=int(self.cfg.head_dim),
            dtype=self.dtype,
            device=self.device,
            frac_vram_utilization=float(self.config.frac_vram_utilization),
        )
        LOG.info(
            "warming NMC prefill batch=1 prompt_tokens=%d chunk_size=%d",
            prompt_len,
            self.config.nmc_prefill_chunk_size,
        )
        warmup_state = None
        try:
            warmup_ids = torch.zeros(
                (1, prompt_len), device=self.device, dtype=torch.long
            )
            warmup_state = torch_prefill._nmc_prefill_forward_from_ids(
                input_ids=warmup_ids,
                weights=self.weights,
                cfg=self.cfg,
                device=self.device,
                dtype=self.dtype,
                page_block=int(self.config.nmc_page_block),
                prefill_chunk_size=int(self.config.nmc_prefill_chunk_size),
                frac_vram_utilization=float(self.config.frac_vram_utilization),
                max_seq=prompt_len + self._kv_reservation_tokens,
                temperature=0.0,
                top_p=1.0,
                trace=None,
            )
            torch.cuda.synchronize(self.device)
        finally:
            if warmup_state is not None:
                warmup_state.close()
        # The fixed-width generated-id buffer must cover the largest possible
        # output. Physical KV remains demand-paged from the shared arena above;
        # do not use this logical output width as the arena reservation floor.
        session_max_new = int(self.config.nmc_max_context)
        LOG.info(
            "NMC prefill warmup completed in %.3fs; creating decode service max_new=%d",
            time.perf_counter() - started,
            session_max_new,
        )
        self._session = NmcDecodeSession(
            weights=self.weights, cfg=self.cfg, device=self.device, dtype=self.dtype,
            session_batch=self.config.bs,
            session_max_new=session_max_new,
            kv_reservation_tokens=self._kv_reservation_tokens,
            max_context=self.config.nmc_max_context,
            num_sms=self.config.nmc_num_sms, page_block=self.config.nmc_page_block,
            prefill_chunk_size=self.config.nmc_prefill_chunk_size,
            frac_vram_utilization=self.config.frac_vram_utilization,
            max_attn_splits=self.config.nmc_max_attn_splits,
            min_attn_chunk=self.config.nmc_min_attn_chunk,
            batched_prefill=self.config.nmc_batched_prefill,
            batched_prefill_max_batch=self.config.nmc_batched_prefill_max_batch,
            batched_prefill_token_budget=self.config.nmc_batched_prefill_token_budget,
            batched_prefill_coalesce_ms=self.config.nmc_batched_prefill_coalesce_ms,
            eos_token_id=self.cfg.eos_token_id, pause_timeout_ms=5000,
            attn_drain=self.config.nmc_attn_drain,
            attn_drain_sms=self.config.nmc_attn_drain_sms)
        if self.config.nmc_batched_prefill:
            LOG.info(
                "warming packed NMC prefill max_batch=%d token_budget=%d",
                self.config.nmc_batched_prefill_max_batch,
                self.config.nmc_batched_prefill_token_budget,
            )
            self._session.warmup_batched_prefill()
        LOG.info(
            "NMC decode service ready in %.3fs batch=%d max_context=%d",
            time.perf_counter() - started,
            self.config.bs,
            self.config.nmc_max_context,
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def encode_prompt_ids(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        import prefill.torch_prefill as torch_prefill

        return torch_prefill.encode_prompt_token_ids(
            tokenizer=self.runner.tokenizer,
            prompt=str(prompt),
            add_special_tokens=add_special_tokens,
        )

    def _submit(
        self,
        prompt: str,
        params: BatchParams,
        *,
        add_special_tokens: bool,
    ):
        if self._session is None:
            raise RuntimeError("session not initialized; warmup did not run")
        # CPU encode: session.submit takes a host list, so ids placed on CUDA
        # here would be copied straight back.
        ids_list = self.encode_prompt_ids(
            prompt, add_special_tokens=add_special_tokens
        )
        return self._session.submit(
            ids_list,
            max_new=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
        )

    def _row_from_ids(
        self,
        ids: Sequence[int],
        finish_reason: str,
        *,
        prompt_token_count: int,
    ) -> GeneratedText:
        ids = [int(x) for x in ids]
        raw = self.runner.tokenizer.decode(ids, skip_special_tokens=False)
        return GeneratedText(
            text=_display_text_from_raw(raw),
            raw_text=raw,
            token_ids=tuple(ids),
            finish_reason=finish_reason,
            prompt_token_count=int(prompt_token_count),
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        params: BatchParams,
        *,
        add_special_tokens: bool,
    ) -> list[GeneratedText]:
        if len(prompts) != 1:
            raise ValueError("session engine expects exactly one prompt per submission")
        req = self._submit(
            prompts[0], params, add_special_tokens=add_special_tokens
        )
        ids = list(req.iter_tokens())
        return [
            self._row_from_ids(
                ids,
                req.finish_reason or "length",
                prompt_token_count=len(req.prompt_ids),
            )
        ]

    def generate_batch_stream(
        self,
        prompts: Sequence[str],
        params: BatchParams,
        callback,
        *,
        add_special_tokens: bool,
    ) -> int:
        if len(prompts) != 1:
            raise ValueError("session engine expects exactly one prompt per submission")
        req = self._submit(
            prompts[0], params, add_special_tokens=add_special_tokens
        )
        prompt_token_count = len(req.prompt_ids)
        try:
            for tok in req.iter_tokens():
                cont = callback([{"token_ids": [int(tok)], "finish_reason": None}])
                if cont is False:
                    if self._session is not None:
                        self._session.cancel(req)
                    break
        finally:
            callback(
                [
                    {
                        "token_ids": [],
                        "finish_reason": req.finish_reason or "length",
                        "prompt_token_count": prompt_token_count,
                    }
                ]
            )
        return prompt_token_count


class SessionBatcher:
    """Async single-request adapter over ``NmcMkSessionEngine``.

    Concurrency lives in the session, so each submit just runs one request on a
    worker thread. No microbatching/padding here.
    """

    def __init__(self, engine: NmcMkSessionEngine) -> None:
        self.engine = engine

    async def submit(
        self,
        prompt: str,
        params: BatchParams,
        *,
        add_special_tokens: bool,
    ) -> GeneratedText:
        def _run() -> GeneratedText:
            rows = self.engine.generate_batch(
                [prompt], params, add_special_tokens=add_special_tokens
            )
            return _normalize_generated_row(rows[0], params.stop)

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        return None


class NmcPromptFormatter:
    """Formats chat messages with the NMC checkpoint tokenizer/template."""

    fast = False

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def format_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
    ) -> str:
        clean_messages = _normalize_chat_messages(messages)
        if not clean_messages:
            raise ValueError("messages must contain at least one item")
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = [tool.raw for tool in tools]
        if tool_choice.mode != "auto":
            kwargs["tool_choice"] = _tool_choice_for_template(tool_choice)
        try:
            rendered = self._tokenizer.apply_chat_template(clean_messages, **kwargs)
        except TypeError as exc:
            raise ValueError("NMC tokenizer chat template does not support requested tool formatting") from exc
        if not isinstance(rendered, str):
            raise TypeError("NMC tokenizer chat template must return text")
        # Keep the template text intact, including its leading BOS. The chat
        # path tokenizes with add_special_tokens=False so we do not add a
        # second BOS or a trailing EOS. Completions still encode raw prompts
        # with add_special_tokens=True.
        return rendered

    def _get_tokenizer(self) -> Any:
        return self._tokenizer


def build_app(
    config: MkServerConfig | None = None,
    *,
    engine: Any | None = None,
    batcher: SessionBatcher | None = None,
    formatter: Any | None = None,
) -> FastAPI:
    if engine is None:
        if config is None:
            config = MkServerConfig()
        engine = NmcMkSessionEngine(config)
    config = engine.config
    require_cohere_melody()
    if batcher is None:
        batcher = SessionBatcher(engine)
    if formatter is None:
        formatter = NmcPromptFormatter(engine.runner.tokenizer)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            warmup_prefill = getattr(engine, "warmup_prefill", None)
            if callable(warmup_prefill):
                # Startup does not yield until compilation and CUDA execution
                # finish, so no request can race the warmup.
                await asyncio.to_thread(warmup_prefill)
            yield
        finally:
            await batcher.close()
            close_engine = getattr(engine, "close", None)
            if callable(close_engine):
                await asyncio.to_thread(close_engine)

    app = FastAPI(
        title="Megakernel NMC OpenAI-compatible text server",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.engine = engine
    app.state.batcher = batcher
    app.state.formatter = formatter

    @app.exception_handler(HTTPException)
    async def _log_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        # Uvicorn's access log only shows "400 Bad Request"; surface the detail
        # so operators can see e.g. unsupported sampling params without curling.
        LOG.warning(
            "%s %s -> %d: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "megakernel-nmc",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: Request) -> dict[str, Any]:
        payload = await _json_payload(request)
        _reject_streaming(payload)
        _validate_model(payload, config.model_id)
        if _request_int(payload.get("n", 1), "n") != 1:
            raise HTTPException(
                status_code=400,
                detail=f"only n=1 is supported; {_SUPPORTED_SAMPLING_SUMMARY}",
            )
        prompts = _completion_prompts(payload.get("prompt"))
        params = _params_from_payload(payload, config)
        LOG.info("completion parameters: %s", params)
        rows = await asyncio.gather(
            *(
                batcher.submit(prompt, params, add_special_tokens=True)
                for prompt in prompts
            )
        )
        choices = [
            {
                "text": row.text,
                "index": idx,
                "logprobs": None,
                "finish_reason": row.finish_reason,
                "token_ids": list(row.token_ids),
            }
            for idx, row in enumerate(rows)
        ]
        prompt_tokens = sum(row.prompt_token_count for row in rows)
        completion_tokens = sum(_completion_token_count(row, params) for row in rows)
        return {
            "id": _response_id("cmpl"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": config.model_id,
            "choices": choices,
            "usage": _usage(prompt_tokens, completion_tokens),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        payload = await _json_payload(request)
        _validate_model(payload, config.model_id)
        if _request_int(payload.get("n", 1), "n") != 1:
            raise HTTPException(
                status_code=400,
                detail=f"only n=1 is supported; {_SUPPORTED_SAMPLING_SUMMARY}",
            )
        tools = _parse_tools(payload.get("tools"))
        tool_choice = _parse_tool_choice(payload.get("tool_choice"), tools)
        messages = _parse_chat_messages(payload.get("messages"), tools)
        tool_policy = _tool_call_policy(tools, tool_choice)
        include_reasoning = _parse_include_reasoning(payload)
        _log_chat_tool_request(tools, tool_choice)
        try:
            prompt = formatter.format_messages(messages, tools=tools, tool_choice=tool_choice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        params = _params_from_payload(payload, config)
        LOG.info("chat completion parameters: %s", params)
        if _request_bool(payload.get("stream", False), "stream"):
            include_usage = _parse_stream_include_usage(payload)
            return StreamingResponse(
                _chat_stream_generator(
                    engine=engine,
                    formatter=formatter,
                    config=config,
                    prompt=prompt,
                    params=params,
                    tool_policy=tool_policy,
                    include_reasoning=include_reasoning,
                    include_usage=include_usage,
                ),
                media_type="text/event-stream",
            )
        row = await batcher.submit(prompt, params, add_special_tokens=False)
        prompt_tokens = row.prompt_token_count
        completion_tokens = _completion_token_count(row, params)
        try:
            message, finish_reason = _nmc_chat_message_from_raw(
                row.raw_text if row.raw_text is not None else row.text,
                row.finish_reason,
                tool_policy=tool_policy,
                include_reasoning=include_reasoning,
                stop=params.stop,
            )

        except ToolCallGenerationError as exc:
            _log_chat_tool_output(None, error=str(exc))
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        _log_chat_tool_output(message, finish_reason=finish_reason)
        return {
            "id": _response_id("chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(prompt_tokens, completion_tokens),
        }

    return app


async def _json_payload(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    return payload


def _request_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{field} must be an integer")
    return int(value)


def _request_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=400, detail=f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HTTPException(status_code=400, detail=f"{field} must be finite")
    return result


def _request_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field} must be a boolean")
    return value


def _params_from_payload(payload: dict[str, Any], config: MkServerConfig) -> BatchParams:
    # OpenAI chat Completions renamed max_tokens -> max_completion_tokens; accept both.
    has_max_tokens = "max_tokens" in payload and payload["max_tokens"] is not None
    has_max_completion = (
        "max_completion_tokens" in payload and payload["max_completion_tokens"] is not None
    )
    if has_max_tokens and has_max_completion:
        legacy_max_tokens = _request_int(payload["max_tokens"], "max_tokens")
        completion_max_tokens = _request_int(
            payload["max_completion_tokens"],
            "max_completion_tokens",
        )
        if legacy_max_tokens != completion_max_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "max_tokens and max_completion_tokens disagree "
                    f"({payload['max_tokens']!r} vs {payload['max_completion_tokens']!r}); "
                    f"{_SUPPORTED_SAMPLING_SUMMARY}"
                ),
            )
        max_tokens = legacy_max_tokens
    elif has_max_completion:
        max_tokens = _request_int(
            payload["max_completion_tokens"],
            "max_completion_tokens",
        )
    elif has_max_tokens:
        max_tokens = _request_int(payload["max_tokens"], "max_tokens")
    else:
        max_tokens = int(config.max_new)
    if max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "max_tokens / max_completion_tokens must be positive; "
                f"{_SUPPORTED_SAMPLING_SUMMARY}"
            ),
        )
    temperature = _request_float(
        payload.get("temperature", config.temperature),
        "temperature",
    )
    if temperature < 0.0:
        raise HTTPException(
            status_code=400,
            detail=f"temperature must be >= 0; {_SUPPORTED_SAMPLING_SUMMARY}",
        )
    top_p = _request_float(payload.get("top_p", 1.0), "top_p")
    if not (0.0 < top_p <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"top_p must be in (0, 1]; {_SUPPORTED_SAMPLING_SUMMARY}",
        )
    if top_p != 1.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"top_p={top_p} is not supported (only top_p=1.0); "
                f"{_SUPPORTED_SAMPLING_SUMMARY}"
            ),
        )
    # top_k=1 is greedy (argmax); -1/0/None mean "disabled". Other top_k
    # values would change the distribution and are not implemented.
    if "top_k" in payload:
        top_k = payload["top_k"]
        if isinstance(top_k, bool) or (
            top_k is not None and not isinstance(top_k, int)
        ):
            raise HTTPException(status_code=400, detail="top_k must be an integer or null")
        if top_k not in (None, 0, -1, 1):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"top_k={top_k!r} is not supported (only top_k=1 / greedy, "
                    f"or unset); {_SUPPORTED_SAMPLING_SUMMARY}"
                ),
            )
        if temperature != 0.0 and top_k == 1:
            raise HTTPException(
                status_code=400,
                detail="temperature should be 0.0 for greedy decode",
            )
    for unsupported in ("frequency_penalty", "presence_penalty", "logit_bias"):
        if unsupported not in payload:
            continue
        value = payload[unsupported]
        # Treat common "unset / disabled" sentinels as absent so OpenAI-style
        # clients that always send defaults are not rejected.
        if value in (None, 0, 0.0, {}):
            continue
        raise HTTPException(
            status_code=400,
            detail=(
                f"{unsupported}={value!r} is not supported; "
                f"{_SUPPORTED_SAMPLING_SUMMARY}"
            ),
        )
    if payload.get("seed") is not None:
        # Not very important feature for now but some eval scripts emit this field. So we print a warning instead of raising an error.
        # raise HTTPException(
        #     status_code=400,
        #     detail=f"per-request seed is not supported; {_SUPPORTED_SAMPLING_SUMMARY}",
        # )
        LOG.warning("per-request seed is not supported; %s", _SUPPORTED_SAMPLING_SUMMARY)
    if payload.get("logprobs") is not None and payload.get("logprobs") is not False:
        raise HTTPException(
            status_code=400,
            detail=f"logprobs are not supported; {_SUPPORTED_SAMPLING_SUMMARY}",
        )
    if payload.get("top_logprobs") is not None:
        raise HTTPException(
            status_code=400,
            detail=f"top_logprobs are not supported; {_SUPPORTED_SAMPLING_SUMMARY}",
        )
    stop = _parse_stop(payload.get("stop"))
    return BatchParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)


def _parse_include_reasoning(payload: dict[str, Any]) -> bool:
    value = payload.get("include_reasoning", True)
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail="include_reasoning must be a boolean")
    return value


def _parse_stream_include_usage(payload: dict[str, Any]) -> bool:
    """Whether to emit a final streaming usage chunk.

    Defaults to True (force-include) so clients that forget
    ``stream_options.include_usage`` still get prompt/completion counts.
    Explicit ``stream_options.include_usage=false`` opts out.
    """
    options = payload.get("stream_options")
    if options is None:
        return True
    if not isinstance(options, dict):
        raise HTTPException(status_code=400, detail="stream_options must be an object")
    if "include_usage" not in options:
        return True
    value = options["include_usage"]
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail="stream_options.include_usage must be a boolean")
    return value


def _parse_stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise HTTPException(status_code=400, detail="stop must be a string or list of strings")


def _completion_prompts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    raise HTTPException(status_code=400, detail="prompt must be a string or non-empty list of strings")


def _reject_streaming(payload: dict[str, Any]) -> None:
    if _request_bool(payload.get("stream", False), "stream"):
        raise HTTPException(status_code=501, detail="streaming is not supported yet")


def _validate_model(payload: dict[str, Any], served_model: str) -> None:
    requested = payload.get("model")
    if requested is not None and requested != served_model:
        raise HTTPException(status_code=404, detail=f"model {requested!r} is not served by this process")


def _normalize_generated_row(row: Any, stop: Sequence[str]) -> GeneratedText:
    if isinstance(row, GeneratedText):
        return _with_stop(row, stop)
    if isinstance(row, dict):
        token_ids_obj = row.get("token_ids", ())
        token_ids = tuple(int(v) for v in token_ids_obj) if isinstance(token_ids_obj, list) else ()
        raw_text_obj = row.get("raw_text")
        raw_text = str(raw_text_obj) if isinstance(raw_text_obj, str) else None
        finish_reason = str(row.get("finish_reason", "length"))
        if finish_reason not in ("length", "stop"):
            finish_reason = "length"
        return _with_stop(
            GeneratedText(
                text=str(row.get("text", "")),
                raw_text=raw_text,
                token_ids=token_ids,
                finish_reason=finish_reason,
                prompt_token_count=int(row.get("prompt_token_count", 0) or 0),
            ),
            stop,
        )
    if isinstance(row, list):
        token_ids = tuple(int(v) for v in row)
        return GeneratedText(text=" ".join(str(v) for v in token_ids), token_ids=token_ids)
    text = str(row)
    match = _ROW_OUTPUT_RE.match(text)
    if match is None:
        return _with_stop(GeneratedText(text=text), stop)
    token_ids: tuple[int, ...] = ()
    with contextlib.suppress(Exception):
        ids_obj = ast.literal_eval(match.group("ids"))
        if isinstance(ids_obj, list):
            token_ids = tuple(int(v) for v in ids_obj)
    rendered = match.group("text")
    with contextlib.suppress(Exception):
        rendered_obj = ast.literal_eval(rendered)
        if isinstance(rendered_obj, str):
            rendered = rendered_obj
    return _with_stop(GeneratedText(text=rendered, token_ids=token_ids), stop)


def _with_stop(row: GeneratedText, stop: Sequence[str]) -> GeneratedText:
    text = _truncate_at_stop(row.text, stop)
    if text == row.text:
        return row
    return GeneratedText(
        text=text,
        raw_text=row.raw_text,
        token_ids=row.token_ids,
        finish_reason="stop",
        prompt_token_count=row.prompt_token_count,
    )


def _truncate_at_stop(text: str, stop: Sequence[str]) -> str:
    cut: int | None = None
    for marker in stop:
        if not marker:
            continue
        idx = text.find(marker)
        if idx >= 0:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]


def _completion_token_count(row: GeneratedText, params: BatchParams) -> int:
    return len(row.token_ids) if row.token_ids else 0


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("text chat content parts require a string text field")
                parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
            else:
                raise ValueError("only text chat content is supported")
        return "".join(parts)
    if content is None:
        return ""
    raise ValueError("only text chat content is supported")


def _parse_chat_messages(value: Any, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    try:
        messages = _normalize_chat_messages(value)
        _validate_tool_message_links(messages)
        _validate_assistant_tool_names(messages, tools)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return messages


def _normalize_chat_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_messages: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{idx}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{idx}].role must be a non-empty string")
        clean: dict[str, Any] = {"role": role, "content": _message_content_to_text(message.get("content", ""))}
        if role == "assistant" and message.get("tool_calls") is not None:
            clean["tool_calls"] = _normalize_assistant_tool_calls(message.get("tool_calls"), f"messages[{idx}]")
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError(f"messages[{idx}].tool_call_id must be a non-empty string")
            clean["tool_call_id"] = tool_call_id
            name = message.get("name")
            if name is not None:
                if not isinstance(name, str) or not name:
                    raise ValueError(f"messages[{idx}].name must be a non-empty string")
                clean["name"] = name
        clean_messages.append(clean)
    if not clean_messages:
        raise ValueError("messages must contain at least one item")
    return clean_messages


def _normalize_assistant_tool_calls(value: Any, prefix: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{prefix}.tool_calls must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        item_prefix = f"{prefix}.tool_calls[{idx}]"
        if not isinstance(item, dict):
            raise TypeError(f"{item_prefix} must be an object")
        tool_call_id = item.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(f"{item_prefix}.id must be a non-empty string")
        if item.get("type") != "function":
            raise ValueError(f"{item_prefix}.type must be 'function'")
        function = item.get("function")
        if not isinstance(function, dict):
            raise TypeError(f"{item_prefix}.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{item_prefix}.function.name must be a non-empty string")
        arguments = function.get("arguments", "{}")
        arguments_json = _arguments_to_json(arguments, f"{item_prefix}.function.arguments")
        normalized.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
            }
        )
    return normalized


def _validate_tool_message_links(messages: Sequence[dict[str, Any]]) -> None:
    unresolved_tool_call_ids: set[str] = set()
    for idx, message in enumerate(messages):
        role = str(message.get("role", "user"))
        if role == "assistant":
            if unresolved_tool_call_ids:
                raise ValueError(f"messages[{idx}].role cannot appear before all assistant tool calls are resolved")
            unresolved_tool_call_ids = {
                str(item["id"])
                for item in message.get("tool_calls", [])
                if isinstance(item, dict) and item.get("type") == "function"
            }
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id", ""))
            if tool_call_id not in unresolved_tool_call_ids:
                raise ValueError(f"messages[{idx}].tool_call_id does not match a prior assistant tool call")
            unresolved_tool_call_ids.remove(tool_call_id)
        elif unresolved_tool_call_ids:
            raise ValueError(f"messages[{idx}].role cannot appear before all assistant tool calls are resolved")
    if unresolved_tool_call_ids:
        raise ValueError("assistant tool calls must be followed by matching tool messages")


def _validate_assistant_tool_names(messages: Sequence[dict[str, Any]], tools: Sequence[ToolSpec]) -> None:
    if not tools:
        return
    tool_names = {tool.name for tool in tools}
    for msg_idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call_idx, tool_call in enumerate(message.get("tool_calls", [])):
            name = tool_call["function"]["name"]
            if name not in tool_names:
                raise ValueError(
                    f"messages[{msg_idx}].tool_calls[{call_idx}].function.name {name!r} is not present in tools"
                )


def _parse_tools(value: Any) -> tuple[ToolSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise HTTPException(status_code=400, detail="tools must be a list")
    tools: list[ToolSpec] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"tools[{idx}] must be an object")
        if item.get("type") != "function":
            raise HTTPException(status_code=400, detail=f"tools[{idx}].type must be 'function'")
        function = item.get("function")
        if not isinstance(function, dict):
            raise HTTPException(status_code=400, detail=f"tools[{idx}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(status_code=400, detail=f"tools[{idx}].function.name must be a non-empty string")
        if name in seen_names:
            raise HTTPException(status_code=400, detail=f"tools[{idx}].function.name {name!r} is duplicated")
        seen_names.add(name)
        description = function.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise HTTPException(status_code=400, detail=f"tools[{idx}].function.description must be a string")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise HTTPException(status_code=400, detail=f"tools[{idx}].function.parameters must be an object")
        raw_function: dict[str, Any] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        if "strict" in function:
            strict = function["strict"]
            if not isinstance(strict, bool):
                raise HTTPException(status_code=400, detail=f"tools[{idx}].function.strict must be a boolean")
            raw_function["strict"] = strict
        tools.append(
            ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                raw={"type": "function", "function": raw_function},
            )
        )
    return tuple(tools)


def _parse_tool_choice(value: Any, tools: Sequence[ToolSpec]) -> ToolChoice:
    if value is None:
        return ToolChoice(mode="auto", function_name=None)
    if isinstance(value, str):
        if value not in ("none", "auto", "required"):
            raise HTTPException(status_code=400, detail="tool_choice must be one of 'none', 'auto', or 'required'")
        if value == "required" and not tools:
            raise HTTPException(status_code=400, detail="tool_choice 'required' requires at least one tool")
        return ToolChoice(mode=value, function_name=None)
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="tool_choice must be a string or object")
    if value.get("type") != "function":
        raise HTTPException(status_code=400, detail="tool_choice.type must be 'function'")
    function = value.get("function")
    if not isinstance(function, dict):
        raise HTTPException(status_code=400, detail="tool_choice.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="tool_choice.function.name must be a non-empty string")
    tool_names = {tool.name for tool in tools}
    if not tools:
        raise HTTPException(status_code=400, detail="tool_choice function requires at least one tool")
    if name not in tool_names:
        raise HTTPException(status_code=400, detail=f"tool_choice.function.name {name!r} is not present in tools")
    return ToolChoice(mode="function", function_name=name)


def _tool_call_policy(tools: Sequence[ToolSpec], tool_choice: ToolChoice) -> ToolCallPolicy:
    allowed_names = frozenset(tool.name for tool in tools)
    if not tools or tool_choice.mode == "none":
        return ToolCallPolicy(mode="none", function_name=None, allowed_names=allowed_names)
    if tool_choice.mode == "function":
        return ToolCallPolicy(mode="function", function_name=tool_choice.function_name, allowed_names=allowed_names)
    return ToolCallPolicy(mode=tool_choice.mode, function_name=None, allowed_names=allowed_names)


def _arguments_to_json(arguments: Any, label: str) -> str:
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be a valid JSON string") from exc
        return arguments
    try:
        return json.dumps(arguments, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _tool_choice_for_template(tool_choice: ToolChoice) -> Any:
    if tool_choice.mode in ("none", "auto", "required"):
        return tool_choice.mode
    if tool_choice.mode == "function" and tool_choice.function_name is not None:
        return {"type": "function", "function": {"name": tool_choice.function_name}}
    return "auto"


def _response_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n"


def _log_chat_tool_request(tools: Sequence[ToolSpec], tool_choice: ToolChoice) -> None:
    tool_names = ",".join(tool.name for tool in tools)
    choice = tool_choice.mode
    if tool_choice.function_name is not None:
        choice = f"{choice}:{tool_choice.function_name}"
    LOG.info(
        "chat tool request tools_available=%s tool_count=%d tool_choice=%s tool_names=%s",
        bool(tools), len(tools), choice, tool_names or "-",
    )


def _log_chat_tool_output(
    message: dict[str, Any] | None,
    *,
    finish_reason: str | None = None,
    error: str | None = None,
) -> None:
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    parsed_calls = tool_calls if isinstance(tool_calls, list) else []
    tool_names = ",".join(
        str(call.get("function", {}).get("name", "-"))
        for call in parsed_calls
        if isinstance(call, dict)
    )
    fields = [
        f"model_tool_calls={bool(parsed_calls)}",
        f"tool_call_count={len(parsed_calls)}",
        f"tool_names={tool_names or '-'}",
    ]
    if finish_reason is not None:
        fields.append(f"finish_reason={finish_reason}")
    if error is not None:
        fields.append(f"tool_call_error={error}")
    LOG.info("chat tool output %s", " ".join(fields))


def _display_text_from_raw(text: str) -> str:
    return re.sub(r"<\|?[A-Z_]+(?:_TOKEN)?\|?>", "", text).strip()


class _StreamingStopFilter:
    """Withhold possible stop prefixes so they never leak across SSE chunks."""

    def __init__(self, stops: Sequence[str]) -> None:
        self._stops = tuple(stop for stop in stops if stop)
        self._buffer = ""
        self.matched = False

    def feed(self, text: str) -> str:
        if self.matched or not text:
            return ""
        if not self._stops:
            return text
        self._buffer += text
        match_positions = [
            position
            for marker in self._stops
            if (position := self._buffer.find(marker)) >= 0
        ]
        if match_positions:
            cut = min(match_positions)
            emitted = self._buffer[:cut]
            self._buffer = ""
            self.matched = True
            return emitted

        keep = 0
        for marker in self._stops:
            max_prefix = min(len(marker) - 1, len(self._buffer))
            for prefix_len in range(max_prefix, 0, -1):
                if marker.startswith(self._buffer[-prefix_len:]):
                    keep = max(keep, prefix_len)
                    break
        if keep == 0:
            emitted = self._buffer
            self._buffer = ""
            return emitted
        emitted = self._buffer[:-keep]
        self._buffer = self._buffer[-keep:]
        return emitted

    def flush(self) -> str:
        if self.matched:
            return ""
        emitted = self._buffer
        self._buffer = ""
        return emitted


def _bounded_queue_put(
    target: queue.Queue[Any],
    item: Any,
    shutdown: threading.Event,
) -> bool:
    while not shutdown.is_set():
        try:
            target.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


def _streaming_tool_delta(
    tool_delta,
    tool_policy: ToolCallPolicy,
) -> dict[str, Any] | None:
    if tool_policy.mode == "none":
        return None
    if tool_delta.name is not None:
        if tool_delta.name not in tool_policy.allowed_names:
            raise ToolCallGenerationError(
                f"model emitted undeclared tool call: {tool_delta.name}"
            )
        if (
            tool_policy.mode == "function"
            and tool_delta.name != tool_policy.function_name
        ):
            raise ToolCallGenerationError(
                f"model emitted {tool_delta.name!r}; expected "
                f"{tool_policy.function_name!r}"
            )

    function: dict[str, Any] = {}
    if tool_delta.name is not None:
        function["name"] = tool_delta.name
    if tool_delta.arguments is not None:
        function["arguments"] = tool_delta.arguments
    if not function:
        return None

    out: dict[str, Any] = {
        "index": int(tool_delta.index),
        "function": function,
    }
    if tool_delta.name is not None:
        out["type"] = "function"
        if tool_delta.id is not None:
            out["id"] = tool_delta.id
    return out


async def _nmc_chat_stream_generator(
    *,
    engine: Any,
    formatter: Any,
    config: MkServerConfig,
    prompt: str,
    params: BatchParams,
    tool_policy: ToolCallPolicy,
    include_reasoning: bool,
    include_usage: bool,
):
    token_queue: queue.Queue[Any] = queue.Queue(
        maxsize=max(8, int(config.stream_chunk_size))
    )
    event_queue: queue.Queue[Any] = queue.Queue(maxsize=32)
    token_done = object()
    event_done = object()
    shutdown = threading.Event()
    decode_cancel = threading.Event()
    prompts = [prompt]
    response_id = _response_id("chatcmpl")
    created = int(time.time())

    def on_token(rows: list[dict[str, Any]]) -> bool:
        if shutdown.is_set() or decode_cancel.is_set():
            return False
        row = rows[0]
        while not shutdown.is_set() and not decode_cancel.is_set():
            try:
                token_queue.put(row, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def decode_worker() -> None:
        try:
            engine.generate_batch_stream(
                prompts, params, on_token, add_special_tokens=False
            )
        except Exception as exc:  # noqa: BLE001 - propagated through the parser stage.
            _bounded_queue_put(token_queue, exc, shutdown)
        finally:
            _bounded_queue_put(token_queue, token_done, shutdown)

    def parser_worker() -> None:
        parser = Cmd4StreamParser()
        tokenizer = formatter._get_tokenizer()
        token_ids: list[int] = []
        # Windowed incremental detokenization (vLLM-style). Instead of decoding
        # the full token_ids prefix on every step (O(N) per token => O(N^2) over
        # a decode, which starves the GPU during long 32K-128K generations), we
        # decode only a small trailing window. prefix_offset..read_offset is the
        # last already-emitted token span, kept as decode context so the newly
        # added ids resolve with correct byte/char boundaries; read_offset..end
        # are the fresh ids whose text we still owe to the stream.
        prefix_offset = 0
        read_offset = 0
        engine_finish_reason = "length"
        prompt_tokens = 0
        stop_filter = _StreamingStopFilter(params.stop)

        def emit_event(event: dict[str, Any]) -> bool:
            return _bounded_queue_put(event_queue, event, shutdown)

        def handle_delta(delta: ParsedAssistantDelta) -> bool:
            if (
                include_reasoning
                and delta.reasoning is not None
                and not emit_event({
                    "kind": "delta",
                    "delta": {"reasoning": delta.reasoning},
                })
            ):
                return False
            if delta.content is not None and not stop_filter.matched:
                content = stop_filter.feed(delta.content)
                if content and not emit_event({
                    "kind": "delta",
                    "delta": {"content": content},
                }):
                    return False
                if stop_filter.matched:
                    decode_cancel.set()
            tool_calls = [
                mapped
                for tool_delta in delta.tool_calls
                if (mapped := _streaming_tool_delta(tool_delta, tool_policy))
                is not None
            ]
            return not tool_calls or emit_event({
                "kind": "delta",
                "delta": {"tool_calls": tool_calls},
            })

        try:
            while not shutdown.is_set():
                try:
                    item = token_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is token_done:
                    break
                if isinstance(item, BaseException):
                    raise item
                if not isinstance(item, dict):
                    raise TypeError("NMC token queue received an invalid item")

                new_ids_obj = item.get("token_ids", ())
                if isinstance(new_ids_obj, list) and new_ids_obj:
                    token_ids.extend(int(token_id) for token_id in new_ids_obj)
                    # Decode only the trailing window. Both decodes start at
                    # prefix_offset so (for prefix-stable, byte-level BPE
                    # tokenizers) new_text is guaranteed to start with prev_text.
                    new_text = tokenizer.decode(
                        token_ids[prefix_offset:],
                        skip_special_tokens=False,
                    )
                    # A trailing replacement char means the last id(s) are an
                    # incomplete UTF-8 sequence: wait for more ids and leave the
                    # offsets unchanged so those ids stay in the decode window.
                    if not new_text.endswith("\ufffd"):
                        prev_text = tokenizer.decode(
                            token_ids[prefix_offset:read_offset],
                            skip_special_tokens=False,
                        )
                        if not new_text.startswith(prev_text):
                            raise RuntimeError(
                                "incremental tokenizer decode changed an emitted prefix"
                            )
                        text_delta = new_text[len(prev_text):]
                        prefix_offset = read_offset
                        read_offset = len(token_ids)
                        if text_delta and not handle_delta(
                            parser.write_decoded(text_delta)
                        ):
                            return

                finish_reason_obj = item.get("finish_reason")
                if finish_reason_obj in ("stop", "length"):
                    engine_finish_reason = str(finish_reason_obj)
                    count_obj = item.get("prompt_token_count")
                    if count_obj is not None:
                        prompt_tokens = int(count_obj)
                    break

            if shutdown.is_set():
                return
            if not handle_delta(parser.flush_partials()):
                return
            final_output = parser.final_output()
            enforced_calls = _enforce_tool_call_policy(
                final_output.tool_calls,
                tool_policy,
            )
            trailing_content = stop_filter.flush()
            if trailing_content and not emit_event({
                "kind": "delta",
                "delta": {"content": trailing_content},
            }):
                return
            finish_reason = (
                "stop"
                if stop_filter.matched
                else "tool_calls"
                if enforced_calls
                else engine_finish_reason
            )
            finish_event: dict[str, Any] = {
                "kind": "finish",
                "finish_reason": finish_reason,
            }
            if include_usage:
                finish_event["usage"] = _usage(prompt_tokens, len(token_ids))
            emit_event(finish_event)
        except Exception as exc:  # noqa: BLE001 - worker boundary reports via event queue.
            decode_cancel.set()
            emit_event({"kind": "error", "error": exc})
        finally:
            _bounded_queue_put(event_queue, event_done, shutdown)

    decode_thread = threading.Thread(
        target=decode_worker,
        name="nmc-stream-decode",
        daemon=True,
    )
    parser_thread = threading.Thread(
        target=parser_worker,
        name="nmc-stream-parser",
        daemon=True,
    )
    decode_thread.start()
    parser_thread.start()

    yield _sse({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": config.model_id,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,
        }],
    })

    try:
        while True:
            item = await asyncio.to_thread(event_queue.get)
            if item is event_done:
                break
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            if kind == "delta":
                yield _sse({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": config.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": item["delta"],
                        "finish_reason": None,
                    }],
                })
            elif kind == "finish":
                yield _sse({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": config.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": item["finish_reason"],
                    }],
                })
                # OpenAI/vLLM: extra final chunk with empty choices + usage.
                usage = item.get("usage")
                if usage is not None:
                    yield _sse({
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": config.model_id,
                        "choices": [],
                        "usage": usage,
                    })
            elif kind == "error":
                error = item.get("error")
                message = str(error)
                _log_chat_tool_output(None, error=message)
                error_type = (
                    "tool_call_generation_error"
                    if isinstance(error, (ToolCallGenerationError, MelodyParseError))
                    else "generation_error"
                )
                yield _sse({
                    "error": {
                        "message": message,
                        "type": error_type,
                    }
                })
        yield _sse("[DONE]")
    finally:
        shutdown.set()
        decode_cancel.set()
        await asyncio.to_thread(decode_thread.join, 1.0)
        await asyncio.to_thread(parser_thread.join, 1.0)


async def _chat_stream_generator(
    *,
    engine: Any,
    formatter: Any,
    config: MkServerConfig,
    prompt: str,
    params: BatchParams,
    tool_policy: ToolCallPolicy,
    include_reasoning: bool,
    include_usage: bool,
):
    async for event in _nmc_chat_stream_generator(
        engine=engine,
        formatter=formatter,
        config=config,
        prompt=prompt,
        params=params,
        tool_policy=tool_policy,
        include_reasoning=include_reasoning,
        include_usage=include_usage,
    ):
        yield event


def _nmc_chat_message_from_raw(
    raw_text: str,
    finish_reason: str,
    *,
    tool_policy: ToolCallPolicy,
    include_reasoning: bool,
    stop: Sequence[str],
) -> tuple[dict[str, Any], str]:
    try:
        parsed = parse_cmd4(raw_text)
    except MelodyParseError as exc:
        raise ToolCallGenerationError(str(exc)) from exc

    parsed_tool_calls = parsed.tool_calls
    if tool_policy.mode == "none" and parsed_tool_calls:
        LOG.warning(
            "NMC model emitted ACTION while tool_choice=none; suppressing tool calls"
        )
    enforced_tool_calls = _enforce_tool_call_policy(parsed_tool_calls, tool_policy)

    content = parsed.content
    if content is not None:
        truncated_content = _truncate_at_stop(content, stop)
        if truncated_content != content:
            content = truncated_content
            finish_reason = "stop"

    # When truncated (or stopped) with reasoning but no final answer, vLLM-style
    # clients expect content="" rather than null so aggregators don't leave
    # message.content as None. Tool-call replies keep content=null.
    if content is None and not enforced_tool_calls:
        content = ""

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if include_reasoning and parsed.reasoning is not None:
        message["reasoning"] = parsed.reasoning
    if enforced_tool_calls:
        message["tool_calls"] = [
            _openai_tool_call(tool_call) for tool_call in enforced_tool_calls
        ]
        finish_reason = "tool_calls"
    return message, finish_reason


def _enforce_tool_call_policy(
    parsed_tool_calls: Sequence[ParsedToolCall],
    tool_policy: ToolCallPolicy,
) -> tuple[ParsedToolCall, ...]:
    calls = tuple(parsed_tool_calls)
    if tool_policy.mode == "none":
        return ()
    if not calls:
        if tool_policy.mode in ("required", "function"):
            raise ToolCallGenerationError("model did not emit required tool calls")
        return ()
    unknown_names = sorted({call.name for call in calls if call.name not in tool_policy.allowed_names})
    if unknown_names:
        raise ToolCallGenerationError(f"model emitted undeclared tool call(s): {', '.join(unknown_names)}")
    if tool_policy.mode == "function":
        expected = tool_policy.function_name
        if len(calls) != 1 or calls[0].name != expected:
            raise ToolCallGenerationError(f"model did not emit exactly one call to required tool {expected!r}")
    return calls


def _openai_tool_call(parsed_tool_call: ParsedToolCall) -> dict[str, Any]:
    return {
        "id": parsed_tool_call.id,
        "type": "function",
        "function": {
            "name": parsed_tool_call.name,
            "arguments": parsed_tool_call.arguments_json,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default=DEFAULT_NMC_MODEL_ID)
    parser.add_argument("--bs", type=int, default=4, choices=SUPPORTED_BATCH_SIZES)
    parser.add_argument(
        "--max-new",
        type=int,
        default=128,
        help="default max_tokens / max_completion_tokens when the request omits "
             "both; also the KV-arena reservation floor per request",
    )
    parser.add_argument("--stream-chunk-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frac-vram-utilization", type=float, default=0.90)
    parser.add_argument("--lib", dest="lib_path", default=None)
    parser.add_argument(
        "--ckpt",
        required=True,
        help="local North Mini Code checkpoint directory",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "bfloat16"],
        default="bf16",
        help="native decode is BF16-only",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--page-block", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--num-sms", type=int, default=132)
    parser.add_argument("--max-attn-splits", type=int, default=32)
    parser.add_argument("--min-attn-chunk", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument(
        "--batched-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pack simultaneous one-chunk prompts into one weight pass (default ON)",
    )
    parser.add_argument("--batched-prefill-max-batch", type=int, default=8)
    parser.add_argument("--batched-prefill-token-budget", type=int, default=8192)
    parser.add_argument("--batched-prefill-coalesce-ms", type=float, default=0.5)
    parser.add_argument("--attn-drain", dest="attn_drain",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="ATTN_DRAIN on full-attention layers (default ON). "
                             "Pass --no-attn-drain for the static A/B arm.")
    parser.add_argument("--attn-drain-sms", type=int, default=None,
                        help="claimer count per full-attention layer "
                             "(default: --num-sms)")
    parser.add_argument("--debug", action="store_true",
                        help="host-side hang diagnostics: write device pointer "
                             "maps under ./dump, and on watchdog expiry take a "
                             "device memory snapshot and park on the hung "
                             "stream for cuda-gdb instead of exiting. Run with "
                             "--num-sms one below the device SM count so the "
                             "snapshot kernel can schedule.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MkServerConfig:
    return MkServerConfig(
        model_id=str(args.model_id),
        bs=int(args.bs),
        max_new=int(args.max_new),
        stream_chunk_size=int(args.stream_chunk_size),
        temperature=float(args.temperature),
        lib_path=args.lib_path,
        frac_vram_utilization=float(args.frac_vram_utilization),
        nmc_ckpt=args.ckpt,
        nmc_dtype=str(args.dtype),
        nmc_device=str(args.device),
        nmc_page_block=int(args.page_block),
        nmc_prefill_chunk_size=int(args.prefill_chunk_size),
        nmc_num_sms=int(args.num_sms),
        nmc_max_attn_splits=int(args.max_attn_splits),
        nmc_min_attn_chunk=int(args.min_attn_chunk),
        nmc_max_context=int(args.max_context),
        nmc_batched_prefill=bool(args.batched_prefill),
        nmc_batched_prefill_max_batch=int(args.batched_prefill_max_batch),
        nmc_batched_prefill_token_budget=int(args.batched_prefill_token_budget),
        nmc_batched_prefill_coalesce_ms=float(args.batched_prefill_coalesce_ms),
        nmc_attn_drain=bool(args.attn_drain),
        nmc_attn_drain_sms=(
            None if args.attn_drain_sms is None else int(args.attn_drain_sms)),
        debug=bool(args.debug),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    config.validate()
    app = build_app(config)
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run(app, host=args.host, port=int(args.port))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
