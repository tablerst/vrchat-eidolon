from __future__ import annotations

import time
from dataclasses import dataclass


def monotonic_ms() -> int:
    """Monotonic clock in milliseconds.

    Use this for latency measurements.
    """

    return int(time.monotonic() * 1000)


def wall_ms() -> int:
    """Wall clock time in milliseconds."""

    return int(time.time() * 1000)


@dataclass(slots=True)
class TurnTtfa:
    """Per-turn timing markers for TTFA instrumentation."""

    turn_id: str
    eos_proxy_ms: int | None = None
    first_audio_delta_ms: int | None = None
    first_audio_played_ms: int | None = None

    def ttfa_ms(self) -> int | None:
        if self.eos_proxy_ms is None or self.first_audio_played_ms is None:
            return None
        return self.first_audio_played_ms - self.eos_proxy_ms

    def ttf_delta_ms(self) -> int | None:
        if self.eos_proxy_ms is None or self.first_audio_delta_ms is None:
            return None
        return self.first_audio_delta_ms - self.eos_proxy_ms
