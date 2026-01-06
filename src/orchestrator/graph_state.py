from __future__ import annotations

import operator
from typing import Any, Annotated
from typing_extensions import TypedDict

from core.types import ToolCall, ToolResult


class AgentState(TypedDict, total=False):
    # Input
    user_text: str

    # PLAN outputs
    plan_assistant_text: str
    tool_calls: list[ToolCall]

    # ACT batching
    pending_tool_calls: list[ToolCall]
    batch_tool_calls: list[ToolCall]

    # Per-branch payload (for Send -> tool_exec)
    tool_call: ToolCall

    # Accumulated tool artifacts
    tool_results: Annotated[list[ToolResult], operator.add]
    tool_messages: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    executed_calls: Annotated[int, operator.add]

    # Final output
    assistant_text: str
