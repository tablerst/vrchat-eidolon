from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Abstract tool call (stable structure across implementations)."""

    id: str
    name: str
    arguments_json: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    content: str | dict[str, Any]
    error: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PlanResult:
    tool_calls: list[ToolCall]
    assistant_text: str = ""
    raw_events: list[Any] | None = None
