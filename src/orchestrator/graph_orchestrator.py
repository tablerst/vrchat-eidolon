from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from core.config import ToolsConfig
from core.types import ToolCall, ToolResult
from observability import bind_context, get_logger, set_state
from observability.ids import new_session_id, new_trace_id
from qwen.client import QwenClient
from tools.mcp_gateway import McpGateway
from tools.mcp_specs import tool_to_openai_spec
from tools.tool_messages import tool_message_from_result

from .graph_state import AgentState


@dataclass(slots=True)
class TurnOutput:
    assistant_text: str
    tool_results: list[ToolResult]


class GraphOrchestrator:
    """LangGraph-based LISTEN → PLAN → ACT → SPEAK → UPDATE orchestrator."""

    def __init__(
        self,
        *,
        qwen: QwenClient,
        tools_cfg: ToolsConfig,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._qwen = qwen
        self._tools_cfg = tools_cfg
        self._mcp_servers = dict(mcp_servers or {})

        self._session_id = new_session_id()
        self._turn_id = 0
        self._log = get_logger("eidolon.orchestrator")

    async def run_turn_text(self, user_text: str) -> TurnOutput:
        self._turn_id += 1
        trace_id = new_trace_id()
        bind_context(trace_id=trace_id, session_id=self._session_id, turn_id=self._turn_id)

        mcp: McpGateway | None = None
        plan_tools: list[dict[str, Any]] | None = None

        if self._mcp_servers:
            mcp = McpGateway(servers=self._mcp_servers, tools_cfg=self._tools_cfg)
            await mcp.load()

            # Dynamic: expose all loaded MCP tools to the model.
            plan_tools = [tool_to_openai_spec(b.tool, name_override=b.model_name) for b in mcp.bindings()]

        graph = self._build_graph(mcp=mcp, plan_tools=plan_tools)

        t0 = time.perf_counter()
        out_state = cast(
            AgentState,
            await graph.ainvoke(
                {
                    "user_text": user_text,
                    "executed_calls": 0,
                    "pending_tool_calls": [],
                    "batch_tool_calls": [],
                    "tool_results": [],
                    "tool_messages": [],
                    "errors": [],
                }
            ),
        )

        self._log.info(
            "turn_done",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            tool_results=len(out_state.get("tool_results", [])),
            assistant_text_len=len(out_state.get("assistant_text", "")),
        )

        return TurnOutput(
            assistant_text=str(out_state.get("assistant_text", "")),
            tool_results=list(out_state.get("tool_results", [])),
        )

    def run_turn_text_sync(self, user_text: str) -> TurnOutput:
        return asyncio.run(self.run_turn_text(user_text))

    def _build_graph(self, *, mcp: McpGateway | None, plan_tools: list[dict[str, Any]] | None):
        tools_cfg = self._tools_cfg
        max_calls = max(0, int(tools_cfg.max_calls_per_turn))
        max_conc = max(1, int(getattr(tools_cfg, "max_concurrency", 5)))

        async def plan_node(state: AgentState) -> dict[str, Any]:
            set_state("PLAN")
            user_text = str(state.get("user_text", ""))

            t0 = time.perf_counter()
            plan = await asyncio.to_thread(
                self._qwen.plan,
                messages=[{"role": "user", "content": user_text}],
                tools=plan_tools,
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

            # Initialize pending queue.
            pending = list(plan.tool_calls)
            if max_calls > 0 and len(pending) > max_calls:
                pending = pending[:max_calls]

            return {
                "plan_assistant_text": plan.assistant_text or "",
                "tool_calls": list(plan.tool_calls),
                "pending_tool_calls": pending,
                "batch_tool_calls": [],
            }

        async def prepare_batch_node(state: AgentState) -> dict[str, Any]:
            set_state("ACT")

            pending: list[ToolCall] = list(state.get("pending_tool_calls", []))
            executed = int(state.get("executed_calls", 0) or 0)

            if max_calls > 0:
                remaining_budget = max(0, max_calls - executed)
            else:
                remaining_budget = len(pending)

            if remaining_budget <= 0:
                if pending:
                    return {
                        "pending_tool_calls": [],
                        "batch_tool_calls": [],
                        "errors": ["tool_calls_truncated_by_max_calls_per_turn"],
                    }
                return {"batch_tool_calls": [], "pending_tool_calls": []}

            batch = pending[: min(max_conc, remaining_budget)]
            rest = pending[len(batch) :]

            return {"batch_tool_calls": batch, "pending_tool_calls": rest}

        def fan_out(state: AgentState):
            batch: list[ToolCall] = list(state.get("batch_tool_calls", []))
            if not batch:
                return "speak"
            return [Send("tool_exec", {"tool_call": tc}) for tc in batch]

        async def tool_exec_node(state: AgentState) -> dict[str, Any]:
            set_state("ACT")
            tc = state.get("tool_call")
            if not isinstance(tc, ToolCall):
                # Defensive: should not happen.
                return {
                    "errors": ["missing_tool_call"],
                }

            if mcp is None:
                # No tools available.
                r = ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    ok=False,
                    content={"text": "tools disabled", "data": {}, "raw": {}, "meta": {"tool_name": tc.name}},
                    error={"type": "tools_disabled", "message": "tools disabled", "details": {}},
                )
            else:
                r = await mcp.call_tool(tool_call_id=tc.id, name=tc.name, arguments=tc.arguments)

            msg = tool_message_from_result(r)
            return {"tool_results": [r], "tool_messages": [msg], "executed_calls": 1}

        async def after_batch_node(state: AgentState) -> dict[str, Any]:
            set_state("UPDATE")
            return {}

        async def speak_node(state: AgentState) -> dict[str, Any]:
            set_state("SPEAK")
            user_text = str(state.get("user_text", ""))
            plan_text = str(state.get("plan_assistant_text", ""))

            tool_msgs = list(state.get("tool_messages", []))
            # Sort for stability.
            tool_msgs.sort(key=lambda m: str(m.get("tool_call_id", "")))

            chunks = await asyncio.to_thread(
                lambda: list(
                    self._qwen.speak(
                        messages=[
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": plan_text},
                            *tool_msgs,
                        ],
                        stream=True,
                        modalities=["text"],
                    )
                )
            )

            text_parts: list[str] = []
            for ev in chunks:
                if ev.get("type") == "text":
                    text_parts.append(str(ev.get("delta") or ""))

            return {"assistant_text": "".join(text_parts)}

        builder = StateGraph(AgentState)
        builder.add_node("plan", plan_node)
        builder.add_node("prepare_batch", prepare_batch_node)
        builder.add_node("tool_exec", tool_exec_node)
        builder.add_node("after_batch", after_batch_node)
        builder.add_node("speak", speak_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "prepare_batch")

        builder.add_conditional_edges("prepare_batch", fan_out, ["tool_exec", "speak"])
        builder.add_edge("tool_exec", "after_batch")
        builder.add_edge("after_batch", "prepare_batch")

        builder.add_edge("speak", END)

        return builder.compile()
