from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StdioMcpServerConfig:
    """Configuration for launching an MCP server over stdio."""

    command: str
    args: list[str]
    env: dict[str, str] | None = None
    timeout_s: float = 5.0


@dataclass(frozen=True, slots=True)
class HttpMcpServerConfig:
    """Configuration for connecting to an MCP server over Streamable HTTP.

    The upstream MCP SDK calls this transport "StreamableHTTP".
    """

    url: str
    timeout_s: float = 10.0
