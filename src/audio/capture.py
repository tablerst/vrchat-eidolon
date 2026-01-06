from __future__ import annotations

import queue
import time
import wave
from pathlib import Path
from typing import Any

from observability.logging import get_logger

from .types import AudioSegment
from .vad import EnergyVadSegmenter, VadParams


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


class MicrophoneAudioCapture(AudioCapture):
    """Capture microphone audio and return utterances segmented by energy VAD.

    This uses `sounddevice.RawInputStream` with dtype=int16. It is designed to
    be simple and robust on Windows (48kHz, mono).
    """

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 48_000,
        channels: int = 1,
        vad: VadParams | None = None,
        block_ms: int = 20,
        read_timeout_s: float = 0.25,
        queue_max_chunks: int = 200,
    ) -> None:
        self._device = device
        self._sr = int(sample_rate)
        self._ch = int(channels)
        self._block_ms = int(block_ms)
        self._read_timeout_s = float(read_timeout_s)

        self._log = get_logger("eidolon.audio")
        self._q: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=int(queue_max_chunks))
        self._segmenter = EnergyVadSegmenter(sample_rate=self._sr, channels=self._ch, params=vad)

        self._sd: Any | None = None
        self._stream: Any | None = None
        self._dropped_chunks = 0

    def _resolve_device(self) -> str | int | None:
        if self._device is None:
            return None
        if isinstance(self._device, int):
            return self._device

        # Resolve by case-insensitive substring match against device names.
        sd = self._sd
        assert sd is not None

        needle = str(self._device).strip().lower()
        if not needle:
            return None

        devices = sd.query_devices()
        matches: list[int] = []
        for idx, d in enumerate(devices):
            try:
                name = str(d.get("name", ""))
                max_in = int(d.get("max_input_channels", 0) or 0)
            except Exception:
                continue
            if max_in <= 0:
                continue
            if needle in name.lower():
                matches.append(idx)

        if not matches:
            raise ValueError(f"input device not found: {self._device!r}")
        return matches[0]

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'sounddevice'. Install it with: uv add sounddevice") from e

        self._sd = sd

        block_frames = max(1, int(self._sr * self._block_ms / 1000))
        bytes_per_frame = self._ch * 2

        def cb(indata: Any, frames: int, _time: Any, status: Any) -> None:
            if status:
                self._log.warning("audio_input_status", status=str(status))

            # Ensure we copy bytes out of the callback buffer.
            chunk = bytes(indata)
            if len(chunk) < int(frames) * bytes_per_frame:
                return

            try:
                self._q.put_nowait((chunk, int(frames)))
            except queue.Full:
                self._dropped_chunks += 1

        self._stream = sd.RawInputStream(
            samplerate=self._sr,
            channels=self._ch,
            dtype="int16",
            blocksize=block_frames,
            device=self._resolve_device(),
            callback=cb,
        )
        self._stream.start()
        self._log.info(
            "audio_input_started",
            sample_rate=self._sr,
            channels=self._ch,
            block_ms=self._block_ms,
            device=self._device,
        )

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def next_segment(self) -> AudioSegment:
        self._ensure_stream()

        # Block until we get a complete utterance.
        last_log = time.monotonic()
        while True:
            try:
                pcm, frames = self._q.get(timeout=self._read_timeout_s)
            except queue.Empty:
                now = time.monotonic()
                if self._dropped_chunks and (now - last_log) >= 2.0:
                    self._log.warning("audio_input_dropped_chunks", dropped=self._dropped_chunks)
                    last_log = now
                continue

            seg = self._segmenter.push(pcm=pcm, frames=frames)
            if seg is not None:
                return seg

