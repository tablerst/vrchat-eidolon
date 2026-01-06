from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from observability.logging import get_logger

from core.config import ToolsConfig
from core.types import ToolResult

from .mcp_naming import McpNameMaps, build_name_maps, parse_model_tool_name, to_model_tool_name
from .runtime import RateLimiter
from .tool_result_codec import normalize_error, payload_from_output, make_payload


@dataclass(frozen=True, slots=True)
class McpToolBinding:
    server_key: str
    raw_name: str
    model_name: str
    tool: Any


class McpGateway:
    """Async MCP gateway built on langchain-mcp-adapters.

    Responsibilities:
    - Load tools from MCP servers defined in config.
    - Apply model-facing name prefixing for multi-server setups.
    - Execute tools with local guardrails (enabled/whitelist/rate-limit).
    - Return normalized ToolResult payloads.
    """

    def __init__(self, *, servers: dict[str, dict[str, Any]], tools_cfg: ToolsConfig) -> None:
        self._servers = dict(servers)
        self._tools_cfg = tools_cfg
        self._log = get_logger("eidolon.mcp")

        self._multi = len(self._servers) > 1
        self._single_server_key = next(iter(self._servers.keys())) if self._servers else ""

        self._maps: McpNameMaps | None = None
        self._bindings_by_model: dict[str, McpToolBinding] = {}
        self._bindings_by_raw: dict[str, list[McpToolBinding]] = {}
        self._limiters: dict[str, RateLimiter] = {}

        self._loaded = False

    @property
    def multi(self) -> bool:
        return self._multi

    @property
    def name_maps(self) -> McpNameMaps:
        if self._maps is None:
            raise RuntimeError("McpGateway not loaded")
        return self._maps

    def bindings(self) -> list[McpToolBinding]:
        return list(self._bindings_by_model.values())

    async def load(self) -> None:
        if self._loaded:
            return
        if not self._servers:
            raise ValueError("mcp.servers is empty")

        # Build stable prefix maps from server_key.
        self._maps = build_name_maps(list(self._servers.keys()))

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'langchain-mcp-adapters'. Install it with: uv add langchain-mcp-adapters"
            ) from e

        bindings_by_model: dict[str, McpToolBinding] = {}
        bindings_by_raw: dict[str, list[McpToolBinding]] = {}

        for server_key, server_cfg in self._servers.items():
            # Normalize transport alias.
            cfg = dict(server_cfg)
            transport = str(cfg.get("transport", ""))
            if transport == "http":
                cfg["transport"] = "streamable_http"

            # We use one-client-per-server to keep an explicit server_key -> tools mapping.
            client = MultiServerMCPClient({server_key: cfg})  # type: ignore[arg-type]
            tools = await client.get_tools()

            for t in tools:
                raw_name = getattr(t, "name", None)
                if not isinstance(raw_name, str) or not raw_name:
                    continue

                model_name = to_model_tool_name(
                    server_key=server_key,
                    tool_name=raw_name,
                    maps=self.name_maps,
                    multi=self._multi,
                )

                if model_name in bindings_by_model:
                    raise RuntimeError(f"duplicate model tool name after prefixing: {model_name!r}")

                b = McpToolBinding(server_key=server_key, raw_name=raw_name, model_name=model_name, tool=t)
                bindings_by_model[model_name] = b
                bindings_by_raw.setdefault(raw_name, []).append(b)

        self._bindings_by_model = bindings_by_model
        self._bindings_by_raw = bindings_by_raw

        # Build rate limiters keyed by model tool name.
        for model_name in self._bindings_by_model.keys():
            cps = self._rate_limit_for(model_name)
            if cps is not None:
                self._limiters[model_name] = RateLimiter(calls_per_second=cps)

        self._loaded = True
        self._log.info(
            "mcp_tools_loaded",
            extra={
                "servers": len(self._servers),
                "tools": len(self._bindings_by_model),
                "multi": self._multi,
            },
        )

    def _rate_limit_for(self, model_tool_name: str) -> float | None:
        # First: exact model name.
        if model_tool_name in self._tools_cfg.rate_limit:
            return float(self._tools_cfg.rate_limit[model_tool_name])

        # Second: raw name (if unique).
        if self._multi:
            try:
                _, raw = parse_model_tool_name(
                    model_tool_name=model_tool_name,
                    maps=self.name_maps,
                    multi=True,
                    single_server_key=self._single_server_key,
                )
            except Exception:
                return None

            if raw in self._tools_cfg.rate_limit:
                return float(self._tools_cfg.rate_limit[raw])

        return None

    def _is_allowed(self, model_tool_name: str) -> tuple[bool, dict[str, Any] | None]:
        if not self._tools_cfg.enabled:
            return False, normalize_error(error_type="tools_disabled", message="tools disabled")

        if not self._tools_cfg.whitelist:
            return True, None

        wl = set(self._tools_cfg.whitelist)
        if model_tool_name in wl:
            return True, None

        if not self._multi:
            return False, normalize_error(error_type="not_allowed", message="not in whitelist")

        # Allow raw-name whitelist only when it is unambiguous.
        try:
            _, raw = parse_model_tool_name(
                model_tool_name=model_tool_name,
                maps=self.name_maps,
                multi=True,
                single_server_key=self._single_server_key,
            )
        except Exception:
            return False, normalize_error(error_type="not_allowed", message="not in whitelist")

        if raw in wl:
            bindings = self._bindings_by_raw.get(raw, [])
            if len(bindings) == 1:
                return True, None
            return False, normalize_error(
                error_type="not_allowed",
                message="raw tool name is ambiguous across servers",
                details={"raw": raw, "candidates": [b.model_name for b in bindings]},
            )

        return False, normalize_error(error_type="not_allowed", message="not in whitelist")

    async def call_tool(self, *, tool_call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        await self.load()

        ok, err = self._is_allowed(name)
        if not ok:
            payload = make_payload(
                text="tool rejected",
                data={},
                raw={},
                meta={"tool_call_id": tool_call_id, "tool_name": name},
            )
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=False, content=payload, error=err)

        limiter = self._limiters.get(name)
        if limiter and not limiter.allow():
            err = normalize_error(error_type="rate_limited", message="rate limited")
            payload = make_payload(
                text="rate limited",
                data={},
                raw={},
                meta={"tool_call_id": tool_call_id, "tool_name": name},
            )
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=False, content=payload, error=err)

        binding = self._bindings_by_model.get(name)
        if binding is None:
            err = normalize_error(error_type="not_found", message="tool not registered")
            payload = make_payload(
                text="tool not found",
                data={},
                raw={},
                meta={"tool_call_id": tool_call_id, "tool_name": name},
            )
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=False, content=payload, error=err)

        meta = {
            "tool_call_id": tool_call_id,
            "tool_name": name,
            "server_key": binding.server_key,
            "raw_tool_name": binding.raw_name,
        }

        try:
            out = await binding.tool.ainvoke(arguments)
            payload = payload_from_output(out, meta=meta)
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=True, content=payload, error=None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            err = normalize_error(
                error_type="mcp_error",
                message=str(e),
                details={"exc": type(e).__name__},
            )
            payload = make_payload(text="tool error", data={}, raw={"exc": repr(e)}, meta=meta)
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=False, content=payload, error=err)
