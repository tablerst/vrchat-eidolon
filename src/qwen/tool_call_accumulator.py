from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from observability.logging import get_logger

from core.types import ToolCall


@dataclass(slots=True)
class _AccumulatedToolCall:
    id: str
    name: str = ""
    arguments: str = ""
    last_json_error: str | None = None


class ToolCallAccumulator:
    """Accumulate streamed tool_call argument fragments.

    Constraints:
    - arguments may be split into multiple chunks; must accumulate until JSON is parseable.
    - do not parse partial JSON for business logic; only probe parseability and record errors.
    """

    def __init__(self) -> None:
        self._calls: dict[str, _AccumulatedToolCall] = {}
        self._log = get_logger()

    def add_delta(self, delta_tool_calls: list[dict[str, Any]]) -> None:
        for tc in delta_tool_calls:
            tc_id = str(tc.get("id") or "")
            if not tc_id:
                # Without id we cannot accumulate.
                continue

            acc = self._calls.setdefault(tc_id, _AccumulatedToolCall(id=tc_id))

            fn = tc.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                acc.name = name

            args = fn.get("arguments")
            if isinstance(args, str) and args:
                acc.arguments += args

            self._probe_json(acc)

    def _probe_json(self, acc: _AccumulatedToolCall) -> None:
        if not acc.arguments:
            return

        try:
            json.loads(acc.arguments)
            acc.last_json_error = None
        except json.JSONDecodeError as e:
            acc.last_json_error = f"{e.msg} (pos={e.pos})"

        self._log.debug(
            "tool_call_accumulate",
            extra={
                "tool_call_id": acc.id,
                "tool_name": acc.name,
                "arguments_len": len(acc.arguments),
                "last_json_error": acc.last_json_error,
            },
        )

    def get_complete_calls(self) -> list[ToolCall]:
        complete: list[ToolCall] = []

        for acc in self._calls.values():
            if not acc.name or not acc.arguments:
                continue
            try:
                parsed = json.loads(acc.arguments)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            complete.append(
                ToolCall(
                    id=acc.id,
                    name=acc.name,
                    arguments_json=acc.arguments,
                    arguments=parsed,
                )
            )

        return complete

    def reset(self) -> None:
        self._calls.clear()
