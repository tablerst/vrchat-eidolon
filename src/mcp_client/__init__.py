"""MCP (Model Context Protocol) client-side integration.

This package intentionally avoids the top-level name `mcp` to prevent shadowing
the upstream MCP Python SDK module (`import mcp`).
"""

from __future__ import annotations

from .client import McpToolClient, StdioMcpToolClient, StreamableHttpMcpToolClient
from .errors import McpClientError, McpDependencyMissingError, McpTimeoutError
from .types import HttpMcpServerConfig, StdioMcpServerConfig

__all__ = [
    "McpClientError",
    "McpDependencyMissingError",
    "McpTimeoutError",
    "McpToolClient",
    "HttpMcpServerConfig",
    "StdioMcpServerConfig",
    "StreamableHttpMcpToolClient",
    "StdioMcpToolClient",
]
