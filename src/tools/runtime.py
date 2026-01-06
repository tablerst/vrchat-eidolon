from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from observability.logging import get_logger

from core.types import ToolResult

from .tool_result_codec import make_payload, normalize_error, payload_from_output


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
        meta = {"tool_call_id": tool_call_id, "tool_name": name}

        if not self._enabled:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="tools disabled", data={}, raw={}, meta=meta),
                error=normalize_error(error_type="tools_disabled", message="tools disabled"),
            )

        if self._whitelist and name not in self._whitelist:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="not in whitelist", data={}, raw={}, meta=meta),
                error=normalize_error(error_type="not_allowed", message="not in whitelist"),
            )

        limiter = self._limiters.get(name)
        if limiter and not limiter.allow():
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="rate limited", data={}, raw={}, meta=meta),
                error=normalize_error(error_type="rate_limited", message="rate limited"),
            )

        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="tool not registered", data={}, raw={}, meta=meta),
                error=normalize_error(error_type="not_found", message="tool not registered"),
            )

        try:
            out = handler(arguments)
            self._log.info("tool_ok", extra={"tool_call_id": tool_call_id, "tool": name})
            return ToolResult(tool_call_id=tool_call_id, name=name, ok=True, content=payload_from_output(out, meta=meta))
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
            error = normalize_error(error_type=e.error_type, message=e.message, details=e.details)
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="tool rejected", data={}, raw={}, meta=meta),
                error=error,
            )
        except Exception as e:  # noqa: BLE001
            self._log.exception("tool_error", extra={"tool_call_id": tool_call_id, "tool": name})
            return ToolResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=False,
                content=make_payload(text="tool exception", data={}, raw={"exc": repr(e)}, meta=meta),
                error=normalize_error(error_type=type(e).__name__, message=str(e), details={"exc": type(e).__name__}),
            )
