from __future__ import annotations

import asyncio
from typing import Any, Protocol

from .errors import McpClientError, McpDependencyMissingError, McpTimeoutError
from .types import HttpMcpServerConfig, StdioMcpServerConfig


class McpToolClient(Protocol):
    """Smallest useful MCP client surface for MVP-0."""

    def call_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        """Call a single MCP tool.

        Implementations may raise McpClientError on failures.
        """
        ...


def _extract_tool_result(payload: Any) -> str | dict[str, Any]:
    """Best-effort extraction from MCP SDK CallToolResult.

    We prefer structured content when available; otherwise we fall back to
    concatenated text blocks; otherwise we return a dict representation.
    """

    structured = getattr(payload, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        return structured

    content = getattr(payload, "content", None) or []
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
        if texts:
            return "\n".join(texts)

    # Last resort: keep something inspectable.
    try:
        # Pydantic v2 models usually have model_dump().
        dump = getattr(payload, "model_dump", None)
        if callable(dump):
            out = dump()
            if isinstance(out, dict):
                return out
    except Exception:  # noqa: BLE001
        pass

    return {"repr": repr(payload)}


class StdioMcpToolClient:
    """MCP client using stdio transport.

    This is implemented as a synchronous wrapper around the upstream MCP Python
    SDK (which is async). For MVP-0 this keeps integration easy with the current
    sync ToolRegistry/orchestrator.

    Note: This creates a new stdio session per call. Long-running sessions can be
    optimized later.
    """

    def __init__(self, cfg: StdioMcpServerConfig) -> None:
        self._cfg = cfg

    def call_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(args, dict):
            raise ValueError("tool args must be a dict")

        try:
            return asyncio.run(self._call_tool_async(name=name, args=args))
        except McpClientError:
            raise
        except Exception as e:  # noqa: BLE001
            raise McpClientError("mcp_error", str(e), details={"exc": type(e).__name__}) from e

    async def _call_tool_async(self, *, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.stdio import (  # type: ignore
                StdioServerParameters,
                stdio_client,
            )
        except Exception as e:  # pragma: no cover
            raise McpDependencyMissingError(
                missing="mcp",
                install_hint="Install it with: uv add mcp",
            ) from e

        server_params = StdioServerParameters(
            command=self._cfg.command,
            args=list(self._cfg.args),
            env=dict(self._cfg.env or {}),
        )

        async def do_call() -> str | dict[str, Any]:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=args)
                    return _extract_tool_result(result)

        try:
            return await asyncio.wait_for(do_call(), timeout=self._cfg.timeout_s)
        except TimeoutError as e:
            raise McpTimeoutError(timeout_s=float(self._cfg.timeout_s)) from e


class StreamableHttpMcpToolClient:
    """MCP client using Streamable HTTP transport.

    This mirrors :class:`StdioMcpToolClient` (sync wrapper around the async MCP SDK)
    and currently creates a new session per call.
    """

    def __init__(self, cfg: HttpMcpServerConfig) -> None:
        self._cfg = cfg

    def call_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(args, dict):
            raise ValueError("tool args must be a dict")

        try:
            return asyncio.run(self._call_tool_async(name=name, args=args))
        except McpClientError:
            raise
        except Exception as e:  # noqa: BLE001
            raise McpClientError("mcp_error", str(e), details={"exc": type(e).__name__}) from e

    async def _call_tool_async(self, *, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        try:
            from mcp.client.session import ClientSession  # type: ignore
            from mcp.client.streamable_http import streamable_http_client  # type: ignore
        except Exception as e:  # pragma: no cover
            raise McpDependencyMissingError(
                missing="mcp",
                install_hint="Install it with: uv add mcp",
            ) from e

        async def do_call() -> str | dict[str, Any]:
            async with streamable_http_client(self._cfg.url) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=args)
                    return _extract_tool_result(result)

        try:
            return await asyncio.wait_for(do_call(), timeout=self._cfg.timeout_s)
        except TimeoutError as e:
            raise McpTimeoutError(timeout_s=float(self._cfg.timeout_s)) from e
