"""MCP client, registry, and tool governance (policy).

Note: The upstream `mcp` package name is reserved by the MCP Python SDK.
We keep our integration layer under a project-specific package name to avoid
import collisions.
"""

from eidolon_mcp.client import build_mcp_client
from eidolon_mcp.policy import (
    PolicyError,
    ToolDisabledError,
    ToolNotAllowedError,
    ToolPolicy,
    ToolRateLimitedError,
)
from eidolon_mcp.registry import McpToolRegistry, ToolResult

__all__ = [
    "build_mcp_client",
    "ToolPolicy",
    "PolicyError",
    "ToolDisabledError",
    "ToolNotAllowedError",
    "ToolRateLimitedError",
    "McpToolRegistry",
    "ToolResult",
]
