"""Cohere Melody cmd4 output parsing for the NMC OpenAI server.

Melody owns structural parsing only. Tool declaration/policy enforcement stays
in serving/server.py, where request context is available.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

try:
    from cohere_melody import PyFilter, PyFilterOptions
except ImportError as exc:
    PyFilter = None  # type: ignore[assignment,misc]
    PyFilterOptions = None  # type: ignore[assignment,misc]
    _MELODY_IMPORT_ERROR: ImportError | None = exc
else:
    _MELODY_IMPORT_ERROR = None


class MelodyParseError(ValueError):
    """Raised when Melody output cannot form a valid assistant response."""


@dataclasses.dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments_json: str


@dataclasses.dataclass(frozen=True)
class ParsedAssistantOutput:
    reasoning: str | None
    content: str | None
    tool_calls: tuple[ParsedToolCall, ...]
    citations: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class StreamingToolCallDelta:
    index: int
    id: str | None
    name: str | None
    arguments: str | None


@dataclasses.dataclass(frozen=True)
class ParsedAssistantDelta:
    reasoning: str | None
    content: str | None
    tool_calls: tuple[StreamingToolCallDelta, ...]


@dataclasses.dataclass
class _StreamingToolState:
    index: int
    id: str
    name: str
    argument_parts: list[str]


def require_cohere_melody() -> None:
    if _MELODY_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "NMC chat parsing requires cohere_melody. Install the pinned server "
        "dependency with: pip install -r requirements.txt"
    ) from _MELODY_IMPORT_ERROR


def _new_cmd4_filter() -> Any:
    require_cohere_melody()
    assert PyFilter is not None
    assert PyFilterOptions is not None
    return PyFilter(PyFilterOptions().cmd4())


def _validated_tool_call(
    tool_call_id: Any,
    name: Any,
    arguments: Any,
) -> ParsedToolCall:
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise MelodyParseError("Melody produced a tool call without an id")
    if not isinstance(name, str) or not name:
        raise MelodyParseError("Melody produced a tool call without a name")
    if not isinstance(arguments, str):
        raise MelodyParseError(f"tool {name!r} produced non-string arguments")
    parsed_id = tool_call_id
    parsed_name = name
    parsed_arguments = arguments
    try:
        arguments_obj = json.loads(parsed_arguments)
    except json.JSONDecodeError as exc:
        raise MelodyParseError(
            f"tool {parsed_name!r} produced invalid JSON arguments: {exc.msg}"
        ) from exc
    if not isinstance(arguments_obj, dict):
        raise MelodyParseError(
            f"tool {parsed_name!r} arguments must decode to a JSON object"
        )
    return ParsedToolCall(
        id=parsed_id,
        name=parsed_name,
        arguments_json=parsed_arguments,
    )


def parse_cmd4(raw_text: str) -> ParsedAssistantOutput:
    result = _new_cmd4_filter().process_full_text(raw_text)
    tool_calls = tuple(
        _validated_tool_call(tool_call.id, tool_call.name, tool_call.arguments)
        for tool_call in result.tool_calls
    )
    return ParsedAssistantOutput(
        reasoning=result.reasoning or None,
        content=result.content or None,
        tool_calls=tool_calls,
        citations=tuple(result.citations),
    )


class Cmd4StreamParser:
    """Stateful Melody parser whose methods return OpenAI-oriented deltas."""

    def __init__(self) -> None:
        self._filter = _new_cmd4_filter()
        self._pending_ids: dict[int, str] = {}
        self._tool_states: dict[int, _StreamingToolState] = {}
        self._reasoning_parts: list[str] = []
        self._content_parts: list[str] = []
        self._citations: list[Any] = []
        self._flushed = False

    def write_decoded(self, text_delta: str) -> ParsedAssistantDelta:
        if self._flushed:
            raise RuntimeError("cannot write to a flushed Melody stream")
        return self._consume_result(self._filter.write_decoded(text_delta))

    def flush_partials(self) -> ParsedAssistantDelta:
        if self._flushed:
            raise RuntimeError("Melody stream was already flushed")
        self._flushed = True
        return self._consume_result(self._filter.flush_partials())

    def final_output(self) -> ParsedAssistantOutput:
        if not self._flushed:
            raise RuntimeError("flush_partials must be called before final_output")
        tool_calls = tuple(
            _validated_tool_call(state.id, state.name, "".join(state.argument_parts))
            for _, state in sorted(self._tool_states.items())
        )
        return ParsedAssistantOutput(
            reasoning="".join(self._reasoning_parts) or None,
            content="".join(self._content_parts) or None,
            tool_calls=tool_calls,
            citations=tuple(self._citations),
        )

    def _consume_result(self, result: Any) -> ParsedAssistantDelta:
        reasoning = result.reasoning or None
        content = result.content or None
        if reasoning is not None:
            self._reasoning_parts.append(reasoning)
        if content is not None:
            self._content_parts.append(content)
        self._citations.extend(result.citations)

        deltas: list[StreamingToolCallDelta] = []
        for tool_call in result.tool_calls:
            index = int(tool_call.index)
            state = self._tool_states.get(index)
            if state is None:
                state = _StreamingToolState(
                    index=index,
                    id="",
                    name="",
                    argument_parts=[],
                )
                self._tool_states[index] = state

            emitted_id: str | None = None
            emitted_name = (
                tool_call.name
                if isinstance(tool_call.name, str) and tool_call.name
                else None
            )
            emitted_arguments = (
                tool_call.arguments
                if isinstance(tool_call.arguments, str) and tool_call.arguments
                else None
            )
            raw_id = (
                tool_call.id
                if isinstance(tool_call.id, str) and tool_call.id
                else ""
            )
            if raw_id:
                state.id = raw_id
                self._pending_ids[index] = raw_id
            if emitted_name is not None:
                state.name = emitted_name
                pending_id = self._pending_ids.pop(index, None)
                emitted_id = raw_id or pending_id
            if emitted_arguments is not None:
                state.argument_parts.append(emitted_arguments)
            if emitted_name is None and emitted_arguments is None:
                continue
            deltas.append(
                StreamingToolCallDelta(
                    index=index,
                    id=emitted_id,
                    name=emitted_name,
                    arguments=emitted_arguments,
                )
            )

        return ParsedAssistantDelta(
            reasoning=reasoning,
            content=content,
            tool_calls=tuple(deltas),
        )
