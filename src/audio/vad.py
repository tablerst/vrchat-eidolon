from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

from .types import AudioSegment


@dataclass(frozen=True, slots=True)
class VadParams:
    """Simple VAD parameters for energy-based segmentation."""

    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 300
    energy_threshold: float = 0.02
    max_utterance_ms: int = 30_000


def rms_energy_int16le(pcm: bytes) -> float:
    """Compute RMS energy for little-endian int16 PCM.

    Returns a normalized value in [0, 1] (approximately), where 1 is full-scale.
    """

    if not pcm:
        return 0.0

    a = array("h")
    a.frombytes(pcm)
    if not a:
        return 0.0

    ss = 0
    for x in a:
        ss += int(x) * int(x)

    mean_sq = ss / float(len(a))
    return math.sqrt(mean_sq) / 32768.0


class EnergyVadSegmenter:
    """Energy-based utterance segmenter.

    This is designed to work with streaming PCM frames. It starts buffering when
    energy exceeds a threshold and ends an utterance after a configured silence.

    NOTE: This is intentionally simple (no ML, no resampling), and works well
    when capture is already configured for 48kHz mono on Windows.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int = 1,
        params: VadParams | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._sr = int(sample_rate)
        self._channels = int(channels)
        self._bytes_per_frame = self._channels * 2
        self._p = params or VadParams()

        self._in_utt = False
        self._buf = bytearray()
        self._frames_total = 0
        self._last_voice_frame = 0

    def push(self, *, pcm: bytes, frames: int) -> AudioSegment | None:
        if frames <= 0:
            return None
        if len(pcm) < frames * self._bytes_per_frame:
            raise ValueError("pcm shorter than frames * channels * 2")

        energy = rms_energy_int16le(pcm)
        is_speech = energy >= float(self._p.energy_threshold)

        silence_frames = int(self._p.silence_duration_ms * self._sr / 1000)
        min_speech_frames = int(self._p.min_speech_duration_ms * self._sr / 1000)
        max_frames = int(self._p.max_utterance_ms * self._sr / 1000)

        if not self._in_utt:
            if not is_speech:
                return None
            self._in_utt = True
            self._buf = bytearray(pcm)
            self._frames_total = int(frames)
            self._last_voice_frame = int(frames)
        else:
            self._buf.extend(pcm)
            self._frames_total += int(frames)
            if is_speech:
                self._last_voice_frame = self._frames_total

        # Stop conditions.
        silence_since_voice = self._frames_total - self._last_voice_frame
        should_stop_by_silence = silence_frames > 0 and silence_since_voice >= silence_frames
        should_stop_by_max = max_frames > 0 and self._frames_total >= max_frames

        if not (should_stop_by_silence or should_stop_by_max):
            return None

        voice_frames = self._last_voice_frame
        if voice_frames < min_speech_frames:
            self.reset()
            return None

        keep_bytes = voice_frames * self._bytes_per_frame
        seg_pcm = bytes(self._buf[:keep_bytes])
        duration_ms = int(voice_frames / self._sr * 1000)

        self.reset()
        return AudioSegment(pcm=seg_pcm, sample_rate=self._sr, duration_ms=duration_ms)

    def flush(self) -> AudioSegment | None:
        """Force-finish the current utterance (if any)."""

        if not self._in_utt:
            return None

        min_speech_frames = int(self._p.min_speech_duration_ms * self._sr / 1000)
        voice_frames = self._last_voice_frame
        if voice_frames < min_speech_frames:
            self.reset()
            return None

        keep_bytes = voice_frames * self._bytes_per_frame
        seg_pcm = bytes(self._buf[:keep_bytes])
        duration_ms = int(voice_frames / self._sr * 1000)

        self.reset()
        return AudioSegment(pcm=seg_pcm, sample_rate=self._sr, duration_ms=duration_ms)

    def reset(self) -> None:
        self._in_utt = False
        self._buf = bytearray()
        self._frames_total = 0
        self._last_voice_frame = 0
