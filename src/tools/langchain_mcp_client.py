from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, cast

from observability.logging import get_logger


class McpToolCallError(RuntimeError):
    """Normalized MCP tool call failure.

    We keep a small, stable set of error types so the rest of the system can
    map failures into ToolResult.error consistently.
    """

    def __init__(self, error_type: str, message: str, *, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class LangchainMcpClientConfig:
    """Configuration for `LangchainMcpClient`.

    `servers` is passed through to `MultiServerMCPClient`.
    """

    servers: dict[str, dict[str, Any]]
    init_timeout_s: float = 10.0


class LangchainMcpClient:
    """Sync wrapper around `langchain-mcp-adapters` for this repo.

    Why a wrapper?
    - The orchestrator + ToolRegistry are synchronous today.
    - `langchain-mcp-adapters` is async-first (`await client.get_tools()`).

    Implementation strategy:
    - Run an asyncio event loop in a dedicated thread.
    - Initialize a MultiServerMCPClient and cache tools by name.
    - Expose a sync `call_tool()` that runs `tool.ainvoke()` on that loop.

    Notes:
    - We intentionally do not expose the full LangChain agent stack here.
    - Connection lifecycle can be improved later via explicit sessions.
    """

    def __init__(self, cfg: LangchainMcpClientConfig) -> None:
        self._cfg = cfg
        self._log = get_logger("eidolon.mcp")

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._closed = False

        self._init_error: BaseException | None = None
        self._tools_by_name: dict[str, Any] = {}

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("MCP client already closed")
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run_loop, name="mcp-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=max(0.1, float(self._cfg.init_timeout_s)))

        if self._init_error is not None:
            raise McpToolCallError(
                "mcp_init_failed",
                str(self._init_error),
                details={"exc": type(self._init_error).__name__},
            ) from self._init_error

        if not self._tools_by_name:
            # Tools may be empty if servers expose none, but that's usually misconfig.
            self._log.warning("mcp_tools_empty")

    def close(self) -> None:
        self._closed = True

        loop = self._loop
        if loop is None:
            return

        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # noqa: BLE001
            pass

    def call_tool(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(args, dict):
            raise ValueError("tool args must be a dict")

        self.start()

        tool = self._tools_by_name.get(name)
        if tool is None:
            raise McpToolCallError("not_found", f"MCP tool not found: {name!r}")

        loop = self._loop
        if loop is None:
            raise McpToolCallError("mcp_unavailable", "MCP loop is not running")

        async def _do() -> Any:
            return await tool.ainvoke(args)

        fut = asyncio.run_coroutine_threadsafe(_do(), loop)
        try:
            return fut.result(timeout=None)
        except TimeoutError as e:
            raise McpToolCallError("timeout", "MCP tool call timed out") from e
        except Exception as e:  # noqa: BLE001
            raise McpToolCallError("mcp_error", str(e), details={"exc": type(e).__name__}) from e

    def tool_names(self) -> list[str]:
        self.start()
        return sorted(self._tools_by_name.keys())

    def _run_loop(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            loop.run_until_complete(self._init_async())
        except BaseException as e:
            self._init_error = e
        finally:
            self._ready.set()
            try:
                if self._loop is not None:
                    self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _init_async(self) -> None:
        # Import inside the thread/loop to keep optional deps isolated.
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'langchain-mcp-adapters'. Install it with: uv add langchain-mcp-adapters"
            ) from e

        # The upstream library types its connection configs, but we intentionally
        # keep our config as a plain JSON-like dict.
        client = MultiServerMCPClient(cast(Any, dict(self._cfg.servers)))
        tools = await client.get_tools()

        tools_by_name: dict[str, Any] = {}
        for t in tools:
            name = getattr(t, "name", None)
            if isinstance(name, str) and name:
                tools_by_name[name] = t

        self._tools_by_name = tools_by_name
        self._log.info("mcp_tools_loaded", extra={"tools": len(self._tools_by_name)})
