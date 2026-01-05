from __future__ import annotations

import re


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def to_model_tool_name(name: str) -> str:
    """Validate and return a model-facing tool name.

    Current project policy:
    - Tool names are already OpenAI-compatible (letters/digits/underscore/hyphen).
    - Dotted tool names are no longer used.
    """

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not _TOOL_NAME_RE.fullmatch(name):
        raise ValueError(f"tool name is not OpenAI-compatible: {name!r}")
    return name


def to_mcp_tool_name(name: str) -> str:
    """Validate and return an MCP tool name.

    Current project policy:
    - MCP tool names are already OpenAI-compatible (same as model-facing names).
    """

    return to_model_tool_name(name)
