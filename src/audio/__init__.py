"""Audio I/O stubs (for parallel development)."""

from __future__ import annotations

from .capture import AudioCapture, MicrophoneAudioCapture, WavFileAudioCapture
from .playback import AudioPlayback, NullAudioPlayback
from .types import AudioSegment
from .vad import EnergyVadSegmenter, VadParams

__all__ = [
    "AudioSegment",
    "AudioCapture",
    "MicrophoneAudioCapture",
    "WavFileAudioCapture",
    "AudioPlayback",
    "NullAudioPlayback",
    "VadParams",
    "EnergyVadSegmenter",
]
