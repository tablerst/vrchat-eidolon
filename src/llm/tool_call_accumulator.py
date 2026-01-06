"""Streaming tool-call argument accumulator.

OpenAI-compatible streaming may deliver tool-call JSON arguments split across chunks.
This module accumulates arguments per tool_call_id until they become valid JSON.

All parsing is best-effort: invalid tool calls should not crash the process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A fully parsed tool call."""

    id: str
    name: str
    args: dict[str, Any]
    raw_args: str


@dataclass(frozen=True)
class InvalidToolCall:
    """A tool call that could not be parsed into JSON args."""

    id: str
    name: str | None
    raw_args: str
    error: str


class ToolCallAccumulator:
    """Accumulate streamed tool-call chunks into parsed ToolCall objects."""

    def __init__(self) -> None:
        self._buffers: dict[str, str] = {}
        self._names: dict[str, str] = {}
        self._seen_order: list[str] = []

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        """Consume a single ToolCallChunk-like dict.

        Expected keys (best-effort):
        - id: str
        - name: str
        - args: str (fragment)
        - index: int
        """

        tool_call_id = chunk.get("id")
        if not tool_call_id:
            idx = chunk.get("index")
            tool_call_id = f"index_{idx}" if idx is not None else "index_unknown"

        if tool_call_id not in self._buffers:
            self._buffers[tool_call_id] = ""
            self._seen_order.append(tool_call_id)

        name = chunk.get("name")
        if name and tool_call_id not in self._names:
            self._names[tool_call_id] = name

        args_fragment = chunk.get("args")
        if isinstance(args_fragment, str) and args_fragment:
            self._buffers[tool_call_id] += args_fragment

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        for ch in chunks:
            if isinstance(ch, dict):
                self.add_chunk(ch)

    def finalize(self) -> tuple[list[ToolCall], list[InvalidToolCall]]:
        """Finalize all known tool calls.

        Returns:
            (tool_calls, invalid_tool_calls)
        """

        tool_calls: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []

        for tool_call_id in self._seen_order:
            raw = self._buffers.get(tool_call_id, "")
            name = self._names.get(tool_call_id)

            try:
                parsed = json.loads(raw) if raw else {}
                if not isinstance(parsed, dict):
                    raise ValueError("tool args must be a JSON object")
                if not name:
                    raise ValueError("missing tool name")

                tool_calls.append(ToolCall(id=tool_call_id, name=name, args=parsed, raw_args=raw))
            except Exception as exc:
                invalid.append(
                    InvalidToolCall(
                        id=tool_call_id,
                        name=name,
                        raw_args=raw,
                        error=str(exc),
                    )
                )

        return tool_calls, invalid
