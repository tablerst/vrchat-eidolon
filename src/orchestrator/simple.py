from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from observability import bind_context, get_logger, set_state
from observability.ids import new_session_id, new_trace_id
from qwen.client import QwenClient
from tools.runtime import ToolRegistry

from core.types import ToolResult


@dataclass(slots=True)
class TurnOutput:
    assistant_text: str
    tool_results: list[ToolResult]


class Orchestrator:
    """Minimal LISTEN → PLAN → ACT → SPEAK → UPDATE orchestrator."""

    def __init__(
        self,
        *,
        qwen: QwenClient,
        tools: ToolRegistry,
        plan_tools: list[dict[str, Any]] | None = None,
        max_calls_per_turn: int = 5,
    ) -> None:
        self._qwen = qwen
        self._tools = tools
        self._plan_tools = plan_tools
        self._max_calls_per_turn = max(0, int(max_calls_per_turn))
        self._session_id = new_session_id()
        self._turn_id = 0
        self._log = get_logger("eidolon.orchestrator")

    def run_turn_text(self, user_text: str) -> TurnOutput:
        self._turn_id += 1
        trace_id = new_trace_id()
        bind_context(trace_id=trace_id, session_id=self._session_id, turn_id=self._turn_id)

        # LISTEN
        set_state("LISTEN")

        # PLAN
        set_state("PLAN")
        t0 = time.perf_counter()
        plan = self._qwen.plan(
            messages=[{"role": "user", "content": user_text}],
            tools=self._plan_tools,
            tool_choice="auto",
            stream=True,
            modalities=["text"],
        )
        self._log.info(
            "plan_done",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            tool_calls=len(plan.tool_calls),
            assistant_text_len=len(plan.assistant_text or ""),
        )

        # ACT
        set_state("ACT")
        tool_results: list[ToolResult] = []
        # Per-turn cap is enforced at the orchestrator boundary.
        tool_calls = plan.tool_calls[:]
        if self._max_calls_per_turn > 0 and len(tool_calls) > self._max_calls_per_turn:
            tool_calls = tool_calls[: self._max_calls_per_turn]
            self._log.warning(
                "tool_calls_truncated",
                requested=len(plan.tool_calls),
                executed=len(tool_calls),
                max_calls_per_turn=self._max_calls_per_turn,
            )

        for tc in tool_calls:
            tool_results.append(self._tools.execute(tool_call_id=tc.id, name=tc.name, arguments=tc.arguments))

        # SPEAK
        set_state("SPEAK")
        tool_msgs: list[dict[str, Any]] = []
        for r in tool_results:
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "name": r.name,
                    "content": r.content if isinstance(r.content, str) else str(r.content),
                }
            )

        chunks = self._qwen.speak(
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": plan.assistant_text or ""},
                *tool_msgs,
            ],
            stream=True,
            modalities=["text"],
        )

        text_parts: list[str] = []
        for ev in chunks:
            if ev.get("type") == "text":
                text_parts.append(str(ev.get("delta") or ""))

        # UPDATE
        set_state("UPDATE")

        return TurnOutput(assistant_text="".join(text_parts), tool_results=tool_results)
