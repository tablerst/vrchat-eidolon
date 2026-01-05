from __future__ import annotations

import pytest

from tools.name_mapping import to_mcp_tool_name, to_model_tool_name


def test_mapping_roundtrip_simple() -> None:
    name = "vrc_chat_send"
    assert to_model_tool_name(name) == name
    assert to_mcp_tool_name(name) == name


def test_mapping_roundtrip_with_underscore_escape() -> None:
    name = "foo_bar-baz"
    assert to_model_tool_name(name) == name
    assert to_mcp_tool_name(name) == name


def test_mapping_rejects_invalid_mcp_chars() -> None:
    with pytest.raises(ValueError):
        to_model_tool_name("vrc/chat/send")


def test_mapping_rejects_invalid_model_chars() -> None:
    with pytest.raises(ValueError):
        to_mcp_tool_name("vrc chat send")
