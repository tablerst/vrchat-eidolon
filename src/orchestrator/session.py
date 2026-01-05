from __future__ import annotations

import time
from dataclasses import dataclass

from observability import bind_context, get_logger, set_state
from observability.ids import new_session_id, new_trace_id
from tools.runtime import ToolRegistry

from core.types import PlanResult


@dataclass(slots=True)
class TurnOutput:
    assistant_text: str


class Session:
    """Minimal state machine skeleton.

    Real audio capture / Qwen calls / tool execution are implemented in their own modules.
    This class pins down the state transitions and observability fields.
    """

    def __init__(self, *, tools: ToolRegistry) -> None:
        self.session_id = new_session_id()
        self.turn_id = 0
        self._log = get_logger()
        self._tools = tools

    def run_turn_text_only(self, user_text: str, *, plan: PlanResult) -> TurnOutput:
        trace_id = new_trace_id()
        self.turn_id += 1

        bind_context(trace_id=trace_id, session_id=self.session_id, turn_id=self.turn_id)

        # PLAN
        set_state("PLAN")
        t0 = time.perf_counter()

        self._log.info(
            "plan_done",
            extra={
                "user_text_len": len(user_text or ""),
                "tool_calls": len(plan.tool_calls),
                "assistant_text_len": len(plan.assistant_text or ""),
            },
        )

        # ACT
        set_state("ACT")
        for tc in plan.tool_calls:
            _ = self._tools.execute(tool_call_id=tc.id, name=tc.name, arguments=tc.arguments)

        # SPEAK (MVP: text only)
        set_state("SPEAK")

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log.info("turn_complete", extra={"latency_ms": round(latency_ms, 2)})

        return TurnOutput(assistant_text=plan.assistant_text or "(no assistant_text)")
