from __future__ import annotations

import wave
from pathlib import Path

from .types import AudioSegment


class AudioCapture:
    """Audio capture interface (VAD can be added later)."""

    def next_segment(self) -> AudioSegment:  # pragma: no cover
        raise NotImplementedError


class WavFileAudioCapture(AudioCapture):
    """Use a WAV file as the input source for offline debugging/testing."""

    def __init__(self, wav_path: str | Path) -> None:
        self._path = Path(wav_path)

    def next_segment(self) -> AudioSegment:
        with wave.open(str(self._path), "rb") as wf:
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        duration_ms = int(n_frames / max(1, framerate) * 1000)
        return AudioSegment(pcm=pcm, sample_rate=framerate, duration_ms=duration_ms)
