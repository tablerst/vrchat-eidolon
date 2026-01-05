from __future__ import annotations

from qwen.tool_call_accumulator import ToolCallAccumulator


def test_toolcall_accumulator_waits_until_json_complete() -> None:
    acc = ToolCallAccumulator()

    acc.add_delta([
        {"id": "1", "function": {"name": "t", "arguments": '{"a":'}},
    ])
    assert acc.get_complete_calls() == []

    acc.add_delta([
        {"id": "1", "function": {"arguments": " 1"}},
    ])
    assert acc.get_complete_calls() == []

    acc.add_delta([
        {"id": "1", "function": {"arguments": "}"}},
    ])

    calls = acc.get_complete_calls()
    assert len(calls) == 1
    assert calls[0].arguments == {"a": 1}
