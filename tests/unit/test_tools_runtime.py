from __future__ import annotations

from tools.runtime import ToolRegistry


def test_tools_disabled() -> None:
    tools = ToolRegistry(enabled=False, whitelist=[], rate_limits={})
    tools.register("ok", lambda args: "done")

    res = tools.execute(tool_call_id="1", name="ok", arguments={})
    assert res.ok is False
    assert res.error and res.error["type"] == "tools_disabled"


def test_tools_whitelist_blocks() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={})
    tools.register("ok", lambda args: "done")

    res = tools.execute(tool_call_id="1", name="nope", arguments={})
    assert res.ok is False
    assert res.error and res.error["type"] == "not_allowed"


def test_tools_rate_limit_blocks() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={"ok": 0.0})
    tools.register("ok", lambda args: "done")

    res = tools.execute(tool_call_id="1", name="ok", arguments={})
    assert res.ok is False
    assert res.error and res.error["type"] == "rate_limited"


def test_tool_not_registered() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={})

    res = tools.execute(tool_call_id="1", name="ok", arguments={})
    assert res.ok is False
    assert res.error and res.error["type"] == "not_found"


def test_tool_exec_ok() -> None:
    tools = ToolRegistry(enabled=True, whitelist=["ok"], rate_limits={})
    tools.register("ok", lambda args: {"echo": args})

    res = tools.execute(tool_call_id="1", name="ok", arguments={"a": 1})
    assert res.ok is True
    assert res.content == {"echo": {"a": 1}}
