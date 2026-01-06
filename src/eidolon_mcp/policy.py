"""Tool governance: allowlist, rate limits, and argument validation.

Policy rules (project defaults):
- If tools are disabled (`tools.enabled=false`), no tool execution is allowed.
- If the allowlist (`tools.whitelist`) is empty, tools are allowed by default (allow-all).
- If the allowlist is non-empty, only listed tool names are allowed.
- Per-tool rate limits are always enforced when configured.

This module must never hard-crash the process: callers should treat policy
violations as expected control flow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyError(RuntimeError):
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


class ToolDisabledError(PolicyError):
    pass


class ToolNotAllowedError(PolicyError):
    pass


class ToolRateLimitedError(PolicyError):
    pass


class RateLimiter:
    """Simple per-tool rate limiter.

    The configured value is interpreted as "calls per second".
    """

    def __init__(self, calls_per_second: float) -> None:
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be > 0")
        self._interval_s = 1.0 / float(calls_per_second)
        self._next_allowed_ts = 0.0

    def check(self, now_ts: float | None = None) -> None:
        now = time.time() if now_ts is None else float(now_ts)
        if now < self._next_allowed_ts:
            raise ToolRateLimitedError(
                f"Rate limited: next allowed at {self._next_allowed_ts:.3f}, now={now:.3f}"
            )
        self._next_allowed_ts = now + self._interval_s


class ToolPolicy:
    """Evaluate whether a tool call is permitted."""

    def __init__(
        self,
        *,
        enabled: bool,
        whitelist: list[str] | None = None,
        rate_limit: dict[str, float] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._whitelist = [t for t in (whitelist or []) if isinstance(t, str) and t]

        self._rate_limiters: dict[str, RateLimiter] = {}
        for name, cps in (rate_limit or {}).items():
            if not isinstance(name, str) or not name:
                continue
            try:
                self._rate_limiters[name] = RateLimiter(float(cps))
            except Exception:
                # Ignore invalid limiter config; callers can decide if they want to validate strictly.
                continue

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ToolPolicy":
        tools = cfg.get("tools")
        if not isinstance(tools, dict):
            tools = {}
        return cls(
            enabled=bool(tools.get("enabled", True)),
            whitelist=tools.get("whitelist") if isinstance(tools.get("whitelist"), list) else [],
            rate_limit=tools.get("rate_limit") if isinstance(tools.get("rate_limit"), dict) else {},
        )

    def check(self, tool_name: str) -> None:
        """Raise a PolicyError if the tool call is not permitted."""

        if not self._enabled:
            raise ToolDisabledError("Tool execution is disabled by configuration")

        if self._whitelist and tool_name not in self._whitelist:
            raise ToolNotAllowedError(f"Tool '{tool_name}' is not in allowlist")

        limiter = self._rate_limiters.get(tool_name)
        if limiter is not None:
            limiter.check()
