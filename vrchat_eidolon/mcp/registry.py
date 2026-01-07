"""Tool registry and unified tool invocation entrypoint.

This module:
- Loads tools from configured MCP server(s).
- Applies a stable naming policy for multi-server setups.
- Invokes tools through LangChain tool objects, guarded by local policy.

Failures are non-fatal: callers receive a structured ToolResult.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from mcp.policy import PolicyError, ToolPolicy


@dataclass(frozen=True)
class ToolResult:
	ok: bool
	content: Any | None = None
	error: str | None = None


def _stable_server_prefix(server_key: str) -> str:
	"""Return a stable 4-char prefix for a server key.

	The first character is guaranteed to be non-numeric.
	"""

	digest = hashlib.sha1(server_key.encode("utf-8")).hexdigest()[:4]
	if digest[0].isdigit():
		digest = chr(ord("a") + int(digest[0])) + digest[1:]
	return digest


class _RenamedTool(BaseTool):
	"""A tool wrapper that overrides the exposed tool name."""

	_inner: BaseTool = PrivateAttr()

	def __init__(self, inner: BaseTool, *, name: str) -> None:
		super().__init__(
			name=name,
			description=getattr(inner, "description", ""),
			args_schema=getattr(inner, "args_schema", None),
		)
		self._inner = inner

	def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
		return self._inner.invoke(input, config=config, **kwargs)

	async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
		return await self._inner.ainvoke(input, config=config, **kwargs)


class McpToolRegistry:
	def __init__(
		self,
		*,
		client: Any,
		policy: ToolPolicy,
		servers: list[str],
		tool_timeout_s: float = 30,
	) -> None:
		self._client = client
		self._policy = policy
		self._servers = servers
		self._tool_timeout_s = float(tool_timeout_s)

		self._tools_by_name: dict[str, BaseTool] = {}

	async def load_tools(self) -> list[BaseTool]:
		"""Load and cache tools from MCP.

		Naming policy:
		- 1 server: keep tool names as-is.
		- N servers: expose tools as `{prefix}__{toolName}` where prefix is a stable 4-char hash.
		"""

		tools: list[BaseTool] = []

		if len(self._servers) <= 1:
			server_name = self._servers[0] if self._servers else None
			loaded = await self._client.get_tools(server_name=server_name)
			tools.extend(loaded)
		else:
			for server_name in self._servers:
				prefix = _stable_server_prefix(server_name)
				loaded = await self._client.get_tools(server_name=server_name)
				for tool in loaded:
					renamed = _RenamedTool(tool, name=f"{prefix}__{tool.name}")
					tools.append(renamed)

		self._tools_by_name = {t.name: t for t in tools}
		return tools

	def get_tool(self, name: str) -> BaseTool | None:
		return self._tools_by_name.get(name)

	async def call_tool(self, *, tool_call_id: str, name: str, args: dict[str, Any]) -> ToolResult:
		"""Invoke a tool by exposed name."""

		try:
			self._policy.check(name)
		except PolicyError as exc:
			return ToolResult(ok=False, error=str(exc))

		tool = self.get_tool(name)
		if tool is None:
			return ToolResult(ok=False, error=f"Unknown tool: {name}")

		payload = {"id": tool_call_id, "name": name, "args": args, "type": "tool_call"}
		try:
			content = await asyncio.wait_for(tool.ainvoke(payload), timeout=self._tool_timeout_s)
			return ToolResult(ok=True, content=content)
		except Exception as exc:
			return ToolResult(ok=False, error=str(exc))
