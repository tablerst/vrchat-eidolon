from __future__ import annotations

from typing import Any, Iterable, Protocol

from .langchain_mcp_client import McpToolCallError
from .runtime import ToolRegistry, ToolRejected


class McpToolClient(Protocol):
    """Minimal interface for calling MCP-backed tools.

    This project intentionally keeps MCP tool execution behind a tiny protocol.
    The concrete implementation is currently backed by `langchain-mcp-adapters`.
    """

    def call_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        ...


def register_mcp_tools(
    registry: ToolRegistry,
    *,
    mcp: McpToolClient,
    tool_names: Iterable[str],
) -> None:
    """Register model-facing tools that delegate to an MCP server.

    Tool names are unified and OpenAI-compatible across the model and MCP server.
    The name mapping layer is used for validation/normalization only.

    The local ToolRegistry remains the first safety boundary (allowlist + rate limit).
    """

    for name in tool_names:
        register_mcp_tool(registry, mcp=mcp, model_tool_name=name)


def register_mcp_tool(registry: ToolRegistry, *, mcp: McpToolClient, model_tool_name: str) -> None:
    """Register a single MCP-backed tool."""

    # Tool names are unified and OpenAI-compatible across model and MCP server.
    mcp_tool_name = model_tool_name

    def handler(args: dict[str, Any]) -> str | dict[str, Any]:
        try:
            return mcp.call_tool(mcp_tool_name, args)
        except McpToolCallError as e:
            # Normalize MCP client failures into structured ToolResult.error.
            raise ToolRejected(e.error_type, e.message, details=e.details) from e

    registry.register(model_tool_name, handler)
