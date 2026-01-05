from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.mcp_bridge import register_mcp_tool
from tools.langchain_mcp_client import McpToolCallError
from tools.runtime import ToolRegistry


@dataclass(slots=True)
class FakeMcpClient:
    last_name: str | None = None
    last_args: dict | None = None
    fail: bool = False

    def call_tool(self, name: str, args: dict) -> dict:
        self.last_name = name
        self.last_args = args
        if self.fail:
            raise McpToolCallError("timeout", "boom", details={"timeout_s": "1"})
        return {"ok": True}


def test_mcp_bridge_calls_mcp_tool_name() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["vrc_chat_send"], rate_limits={})
    mcp = FakeMcpClient()

    register_mcp_tool(tools, mcp=mcp, model_tool_name="vrc_chat_send")

    res = tools.execute(tool_call_id="1", name="vrc_chat_send", arguments={"text": "hi"})
    assert res.ok is True

    assert mcp.last_name == "vrc_chat_send"
    assert mcp.last_args == {"text": "hi"}


def test_mcp_bridge_normalizes_errors() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["vrc_chat_send"], rate_limits={})
    mcp = FakeMcpClient(fail=True)

    register_mcp_tool(tools, mcp=mcp, model_tool_name="vrc_chat_send")

    res = tools.execute(tool_call_id="1", name="vrc_chat_send", arguments={"text": "hi"})
    assert res.ok is False
    assert res.error is not None
    assert res.error["type"] == "timeout"
    assert "boom" in res.error["message"]


def test_mcp_bridge_respects_whitelist() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={})
    mcp = FakeMcpClient()

    register_mcp_tool(tools, mcp=mcp, model_tool_name="vrc_chat_send")

    res = tools.execute(tool_call_id="1", name="vrc_chat_send", arguments={"text": "hi"})
    assert res.ok is False
    assert res.error and res.error["type"] == "not_allowed"
