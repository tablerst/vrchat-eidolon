from __future__ import annotations

import pytest

from tools.mcp_naming import build_name_maps, parse_model_tool_name, to_model_tool_name


def test_single_server_no_prefix() -> None:
    maps = build_name_maps(["vrchat"])
    name = to_model_tool_name(server_key="vrchat", tool_name="vrc_status", maps=maps, multi=False)
    assert name == "vrc_status"


def test_multi_server_prefix_and_roundtrip() -> None:
    maps = build_name_maps(["a", "b"])  # stable sample keys

    model_name = to_model_tool_name(server_key="a", tool_name="vrc_status", maps=maps, multi=True)
    assert "__" in model_name

    prefix, raw = model_name.split("__", 1)
    assert len(prefix) == 4
    assert prefix[0].isalpha()  # first char must not be a digit
    assert raw == "vrc_status"

    server_key, tool_name = parse_model_tool_name(
        model_tool_name=model_name,
        maps=maps,
        multi=True,
        single_server_key="a",
    )
    assert server_key == "a"
    assert tool_name == "vrc_status"


def test_parse_rejects_missing_separator() -> None:
    maps = build_name_maps(["a", "b"])
    with pytest.raises(ValueError):
        parse_model_tool_name(model_tool_name="vrc_status", maps=maps, multi=True, single_server_key="a")
