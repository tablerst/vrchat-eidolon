from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from observability.logging import get_logger

from core.types import ToolResult


class ToolRejected(RuntimeError):
    """Structured tool rejection.

    Use this when a tool handler wants to fail with a normalized error type
    rather than raising an arbitrary exception.
    """

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


@dataclass(slots=True)
class RateLimiter:
    """Very small per-tool rate limiter (simplified token bucket)."""

    calls_per_second: float
    burst: float | None = None
    _tokens: float = 0.0
    _last: float = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if self._last == 0.0:
            self._last = now
            self._tokens = self.burst or self.calls_per_second

        elapsed = max(0.0, now - self._last)
        self._last = now

        cap = self.burst or self.calls_per_second
        self._tokens = min(cap, self._tokens + elapsed * self.calls_per_second)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


ToolHandler = Callable[[dict[str, Any]], str | dict[str, Any]]


class ToolRegistry:
    def __init__(
        self,
        *,
        enabled: bool,
        whitelist: list[str],
        rate_limits: dict[str, float],
    ) -> None:
        self._enabled = enabled
        self._whitelist = set(whitelist)
        self._handlers: dict[str, ToolHandler] = {}
        self._log = get_logger()
        self._limiters: dict[str, RateLimiter] = {
            name: RateLimiter(calls_per_second=cps) for name, cps in rate_limits.items()
        }

    def register(self, name: str, handler: ToolHandler) -> None:
        self._handlers[name] = handler

    def execute(self, *, tool_call_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        if not self._enabled:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具系统已禁用",
                error={"type": "tools_disabled", "message": "tools disabled"},
            )

        if self._whitelist and name not in self._whitelist:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具不在白名单",
                error={"type": "not_allowed", "message": "not in whitelist"},
            )

        limiter = self._limiters.get(name)
        if limiter and not limiter.allow():
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具触发速率限制",
                error={"type": "rate_limited", "message": "rate limited"},
            )

        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具未注册",
                error={"type": "not_found", "message": "tool not registered"},
            )

        try:
            out = handler(arguments)
            self._log.info("tool_ok", extra={"tool_call_id": tool_call_id, "tool": name})
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=True, content=out)
        except ToolRejected as e:
            # Normalized tool failure.
            self._log.info(
                "tool_rejected",
                extra={
                    "tool_call_id": tool_call_id,
                    "tool": name,
                    "error_type": e.error_type,
                },
            )
            error: dict[str, str] = {"type": e.error_type, "message": e.message}
            if e.details:
                # Keep this flat and JSON-friendly.
                error.update({f"detail_{k}": v for k, v in e.details.items()})
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具执行失败",
                error=error,
            )
        except Exception as e:  # noqa: BLE001
            self._log.exception("tool_error", extra={"tool_call_id": tool_call_id, "tool": name})
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content="工具执行异常",
                error={"type": type(e).__name__, "message": str(e)},
            )
