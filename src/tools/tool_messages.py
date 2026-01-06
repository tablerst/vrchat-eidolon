from __future__ import annotations

from typing import Any

from core.types import ToolResult

from .tool_result_codec import dumps_payload


def tool_message_from_result(r: ToolResult) -> dict[str, Any]:
    """Build an OpenAI-compatible tool message from a ToolResult.

    Note:
    - OpenAI tool message content should be a JSON string.
    - We keep `name` for compatibility with current Qwen/OpenAI-compat usage.
    """

    return {
        "role": "tool",
        "tool_call_id": r.tool_call_id,
        "name": r.name,
        "content": dumps_payload(r.content),
    }
