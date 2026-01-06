from __future__ import annotations

from llm.tool_call_accumulator import ToolCallAccumulator


def test_accumulator_parses_fragmented_json_args() -> None:
    acc = ToolCallAccumulator()

    acc.add_chunk({"id": "call_1", "index": 0, "name": "echo", "args": "{\"text\": \"hi\""})
    acc.add_chunk({"id": "call_1", "index": 0, "name": "echo", "args": "}"})

    tool_calls, invalid = acc.finalize()

    assert invalid == []
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "echo"
    assert tool_calls[0].args == {"text": "hi"}


def test_accumulator_keeps_invalid_json_non_fatal() -> None:
    acc = ToolCallAccumulator()

    acc.add_chunk({"id": "call_1", "index": 0, "name": "echo", "args": "{\"text\":"})

    tool_calls, invalid = acc.finalize()

    assert tool_calls == []
    assert len(invalid) == 1
    assert invalid[0].id == "call_1"
    assert invalid[0].name == "echo"
    assert invalid[0].raw_args.startswith("{\"text\":")
