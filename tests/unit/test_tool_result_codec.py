from __future__ import annotations

import json

from core.types import ToolResult
from tools.tool_messages import tool_message_from_result
from tools.tool_result_codec import dumps_payload, make_payload, payload_from_output


def test_payload_from_output_dict() -> None:
    payload = payload_from_output({"ok": True}, meta={"tool_name": "t"})
    assert set(payload.keys()) == {"text", "data", "raw", "meta"}
    assert payload["data"] == {"ok": True}
    assert payload["meta"]["tool_name"] == "t"


def test_dumps_payload_is_json() -> None:
    payload = make_payload(text="hi", data={"a": 1}, raw={"a": 1}, meta={"m": 1})
    s = dumps_payload(payload)
    obj = json.loads(s)
    assert obj["data"]["a"] == 1


def test_tool_message_content_roundtrip() -> None:
    payload = make_payload(text="hi", data={"a": 1}, raw={"a": 1}, meta={"m": 1})
    r = ToolResult(tool_call_id="1", name="t", ok=True, content=payload, error=None)

    msg = tool_message_from_result(r)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "1"
    assert msg["name"] == "t"

    decoded = json.loads(msg["content"])
    assert decoded["data"]["a"] == 1
    assert set(decoded.keys()) == {"text", "data", "raw", "meta"}
