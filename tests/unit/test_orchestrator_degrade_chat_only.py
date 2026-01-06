from __future__ import annotations

from typing import Any

import pytest

from core.config import ToolsConfig
from core.types import PlanResult
from orchestrator.graph_orchestrator import GraphOrchestrator
from qwen.client import FakeQwenClient


class RecordingQwen(FakeQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.last_plan_tools: list[dict[str, Any]] | None = None
        self.last_plan_tool_choice: str | dict[str, Any] | None = None

    def plan(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = True,
        modalities: list[str] | None = None,
    ) -> PlanResult:
        _ = (messages, stream, modalities)
        self.last_plan_tools = tools
        self.last_plan_tool_choice = tool_choice
        return PlanResult(tool_calls=[], assistant_text="(test) planning", raw_events=None)

    def speak(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = True,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
    ):
        _ = (messages, stream, modalities, audio)
        yield {"type": "text", "delta": "(test) reply"}


def test_mcp_load_failure_degrades_to_chat_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingMcpGateway:
        def __init__(self, *, servers: dict[str, dict[str, Any]], tools_cfg: ToolsConfig) -> None:
            _ = (servers, tools_cfg)

        async def load(self) -> None:
            raise RuntimeError("mcp offline")

    monkeypatch.setattr("orchestrator.graph_orchestrator.McpGateway", FailingMcpGateway)

    qwen = RecordingQwen()
    orch = GraphOrchestrator(
        qwen=qwen,
        tools_cfg=ToolsConfig(enabled=True, whitelist=[], rate_limit={}),
        mcp_servers={"server": {"transport": "http", "url": "http://127.0.0.1:9999"}},
    )

    out = orch.run_turn_text_sync("hi")

    assert out.assistant_text.startswith("注意：MCP 工具服务当前不可用")
    assert "(test) reply" in out.assistant_text

    # In degraded mode, planning must be forced to chat-only.
    assert qwen.last_plan_tools is None
    assert qwen.last_plan_tool_choice == "none"


def test_no_mcp_config_runs_chat_only_without_notice() -> None:
    qwen = RecordingQwen()
    orch = GraphOrchestrator(
        qwen=qwen,
        tools_cfg=ToolsConfig(enabled=True, whitelist=[], rate_limit={}),
        mcp_servers=None,
    )

    out = orch.run_turn_text_sync("hi")

    assert not out.assistant_text.startswith("注意：MCP 工具服务当前不可用")
    assert "(test) reply" in out.assistant_text

    assert qwen.last_plan_tools is None
    assert qwen.last_plan_tool_choice == "none"
