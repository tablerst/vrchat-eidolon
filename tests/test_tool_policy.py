from __future__ import annotations

import time

import pytest

from eidolon_mcp.policy import (
    ToolDisabledError,
    ToolNotAllowedError,
    ToolPolicy,
    ToolRateLimitedError,
)


def test_policy_allow_all_when_whitelist_empty() -> None:
    policy = ToolPolicy(enabled=True, whitelist=[], rate_limit={})
    policy.check("any_tool")


def test_policy_blocks_when_disabled() -> None:
    policy = ToolPolicy(enabled=False, whitelist=[], rate_limit={})
    with pytest.raises(ToolDisabledError):
        policy.check("any_tool")


def test_policy_blocks_when_whitelist_non_empty() -> None:
    policy = ToolPolicy(enabled=True, whitelist=["allowed"], rate_limit={})

    policy.check("allowed")
    with pytest.raises(ToolNotAllowedError):
        policy.check("denied")


def test_policy_rate_limit_calls_per_second() -> None:
    policy = ToolPolicy(enabled=True, whitelist=[], rate_limit={"tool": 2})

    # First call ok
    policy.check("tool")

    # Second call immediately should be rate limited.
    with pytest.raises(ToolRateLimitedError):
        policy.check("tool")

    # After 0.5s (2 calls/sec), should be ok again.
    time.sleep(0.51)
    policy.check("tool")
