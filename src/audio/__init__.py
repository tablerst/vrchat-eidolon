"""Audio I/O stubs (for parallel development)."""

from __future__ import annotations

from .capture import AudioCapture, WavFileAudioCapture
from .playback import AudioPlayback, NullAudioPlayback
from .types import AudioSegment

__all__ = [
    "AudioSegment",
    "AudioCapture",
    "WavFileAudioCapture",
    "AudioPlayback",
    "NullAudioPlayback",
]
