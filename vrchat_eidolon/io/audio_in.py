from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import AsyncIterator

import sounddevice as sd
import asyncio


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioInputConfig:
    device: str | int | None
    sample_rate: int
    channels: int
    chunk_ms: int = 100
    queue_max_chunks: int = 20


class AudioInput:
    """Microphone capture using PortAudio (sounddevice).

    Produces raw PCM16LE bytes in ~chunk_ms chunks.

    Notes:
    - The audio callback must stay extremely light. It only enqueues bytes.
    - We use a thread-safe queue for callback -> asyncio handoff.
    """

    def __init__(self, cfg: AudioInputConfig):
        self._cfg = cfg
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=cfg.queue_max_chunks)
        self._stream: sd.RawInputStream | None = None
        self._dropped = 0
        self._effective_sample_rate: int | None = None

    @property
    def device(self) -> str | int | None:
        return self._cfg.device

    @property
    def sample_rate(self) -> int:
        return self._effective_sample_rate or self._cfg.sample_rate

    @property
    def channels(self) -> int:
        return self._cfg.channels

    def start(self) -> None:
        blocksize = int(self._cfg.sample_rate * self._cfg.chunk_ms / 1000)

        def _callback(indata: bytes, frames: int, time_info, status: sd.CallbackFlags) -> None:  # noqa: ANN001
            if status:
                # Avoid logging on every callback; it is too expensive.
                pass
            try:
                self._q.put_nowait(indata)
            except queue.Full:
                self._dropped += 1
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._q.put_nowait(indata)
                except queue.Full:
                    return

        self._stream = sd.RawInputStream(
            device=self._cfg.device,
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype="int16",
            blocksize=blocksize,
            callback=_callback,
        )
        self._stream.start()

        # PortAudio / host API may negotiate a different samplerate than requested.
        # Expose it to downstream resamplers to avoid pitch shift.
        try:
            self._effective_sample_rate = int(getattr(self._stream, "samplerate"))
        except Exception:  # noqa: BLE001
            self._effective_sample_rate = None

        logger.info(
            "audio_in_started",
            extra={
                "device": self._cfg.device,
                "sample_rate": self._cfg.sample_rate,
                "effective_sample_rate": self.sample_rate,
                "channels": self._cfg.channels,
                "chunk_ms": self._cfg.chunk_ms,
            },
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._stream = None

        if self._dropped:
            logger.warning("audio_in_dropped_chunks", extra={"dropped": self._dropped})

        logger.info("audio_in_stopped")
        self._effective_sample_rate = None

    async def __aenter__(self) -> "AudioInput":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    async def chunks(self) -> AsyncIterator[bytes]:
        """Async iterator of pcm16 chunks."""

        while True:
            # Offload the blocking queue.get() to a worker thread.
            data = await asyncio.to_thread(self._q.get)
            yield data
