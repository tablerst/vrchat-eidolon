from __future__ import annotations

from tools.runtime import ToolRegistry


def test_tool_whitelist_blocks() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={})
    tools.register("ok", lambda args: {"ok": True, "args": args})

    res = tools.execute(tool_call_id="1", name="nope", arguments={})
    assert res.ok is False
    assert res.error and res.error["type"] == "not_allowed"


def test_tool_rate_limit_blocks() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={"ok": 0.0})
    tools.register("ok", lambda args: "done")

    res = tools.execute(tool_call_id="1", name="ok", arguments={})

    assert res.ok is False
    assert res.error and res.error["type"] == "rate_limited"
