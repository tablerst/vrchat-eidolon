from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioSegment:
    pcm: bytes
    sample_rate: int
    duration_ms: int
