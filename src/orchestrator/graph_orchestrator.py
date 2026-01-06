from __future__ import annotations

import asyncio
import base64
import io
import time
import wave
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

from audio.types import AudioSegment

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
        degraded_notice: str = ""

        if self._mcp_servers:
            mcp = McpGateway(servers=self._mcp_servers, tools_cfg=self._tools_cfg)
            try:
                await mcp.load()
            except Exception as e:  # noqa: BLE001
                # Degrade to chat-only when MCP is unavailable.
                self._log.warning(
                    "mcp_unavailable_degraded_to_chat_only",
                    exc=type(e).__name__,
                    error_message=str(e),
                )
                mcp = None
                plan_tools = None
                degraded_notice = "注意：MCP 工具服务当前不可用，已自动降级为纯聊天模式（本轮不会调用任何工具，也无法驱动角色动作）。\n\n"
            else:
                # Dynamic: expose loaded MCP tools to the model (only if tools are enabled).
                if self._tools_cfg.enabled:
                    plan_tools = [tool_to_openai_spec(b.tool, name_override=b.model_name) for b in mcp.bindings()]
                else:
                    plan_tools = None

        graph = self._build_graph(mcp=mcp, plan_tools=plan_tools, degraded_notice=degraded_notice)

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

    async def run_turn_audio(self, *, audio: AudioSegment, channels: int = 1, user_text: str = "") -> TurnOutput:
        """Run a single turn with audio+text input.

        The audio is wrapped into a base64 Data URL WAV payload using the OpenAI-compatible
        `input_audio` message part format.
        """

        self._turn_id += 1
        trace_id = new_trace_id()
        bind_context(trace_id=trace_id, session_id=self._session_id, turn_id=self._turn_id)

        wav_bytes = self._pcm16le_to_wav_bytes(
            pcm=audio.pcm,
            sample_rate=int(audio.sample_rate),
            channels=int(channels),
        )
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_url = f"data:;base64,{b64}"

        user_message: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": data_url, "format": "wav"}},
                {"type": "text", "text": str(user_text)},
            ],
        }

        mcp: McpGateway | None = None
        plan_tools: list[dict[str, Any]] | None = None
        degraded_notice: str = ""

        if self._mcp_servers:
            mcp = McpGateway(servers=self._mcp_servers, tools_cfg=self._tools_cfg)
            try:
                await mcp.load()
            except Exception as e:  # noqa: BLE001
                self._log.warning(
                    "mcp_unavailable_degraded_to_chat_only",
                    exc=type(e).__name__,
                    error_message=str(e),
                )
                mcp = None
                plan_tools = None
                degraded_notice = "注意：MCP 工具服务当前不可用，已自动降级为纯聊天模式（本轮不会调用任何工具，也无法驱动角色动作）。\n\n"
            else:
                if self._tools_cfg.enabled:
                    plan_tools = [tool_to_openai_spec(b.tool, name_override=b.model_name) for b in mcp.bindings()]
                else:
                    plan_tools = None

        graph = self._build_graph(mcp=mcp, plan_tools=plan_tools, degraded_notice=degraded_notice)

        t0 = time.perf_counter()
        out_state = cast(
            AgentState,
            await graph.ainvoke(
                {
                    "user_text": str(user_text),
                    "user_message": user_message,
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

    def run_turn_audio_sync(self, *, audio: AudioSegment, channels: int = 1, user_text: str = "") -> TurnOutput:
        return asyncio.run(self.run_turn_audio(audio=audio, channels=channels, user_text=user_text))

    @staticmethod
    def _pcm16le_to_wav_bytes(*, pcm: bytes, sample_rate: int, channels: int) -> bytes:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(int(channels))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm)
        return bio.getvalue()

    def _build_graph(self, *, mcp: McpGateway | None, plan_tools: list[dict[str, Any]] | None, degraded_notice: str):
        tools_cfg = self._tools_cfg
        max_calls = max(0, int(tools_cfg.max_calls_per_turn))
        max_conc = max(1, int(getattr(tools_cfg, "max_concurrency", 5)))

        async def plan_node(state: AgentState) -> dict[str, Any]:
            set_state("PLAN")
            user_text = str(state.get("user_text", ""))

            user_msg = state.get("user_message")
            if isinstance(user_msg, dict) and user_msg.get("role") == "user":
                messages: list[dict[str, Any]] = [dict(user_msg)]
            else:
                messages = [{"role": "user", "content": user_text}]

            tools_for_plan = plan_tools if plan_tools else None
            tool_choice: str | dict[str, Any]
            if tools_for_plan:
                tool_choice = "auto"
            else:
                # Force chat-only planning when no tools are available.
                tool_choice = "none"

            t0 = time.perf_counter()
            plan = await asyncio.to_thread(
                self._qwen.plan,
                messages=messages,
                tools=tools_for_plan,
                tool_choice=tool_choice,
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

            user_msg = state.get("user_message")
            if isinstance(user_msg, dict) and user_msg.get("role") == "user":
                first_user = dict(user_msg)
            else:
                first_user = {"role": "user", "content": user_text}

            tool_msgs = list(state.get("tool_messages", []))
            # Sort for stability.
            tool_msgs.sort(key=lambda m: str(m.get("tool_call_id", "")))

            chunks = await asyncio.to_thread(
                lambda: list(
                    self._qwen.speak(
                        messages=[
                            first_user,
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

            return {"assistant_text": f"{degraded_notice}{''.join(text_parts)}"}

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
