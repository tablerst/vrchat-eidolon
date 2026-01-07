from __future__ import annotations

import asyncio
import base64
import logging
import threading
from dataclasses import dataclass

import sounddevice as sd


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioOutputConfig:
    device: str | int | None
    sample_rate: int
    channels: int


def _pcm24le_to_int32le(pcm24: bytes) -> bytes:
    """Convert PCM24LE packed bytes to PCM32LE bytes with sign extension."""

    if len(pcm24) % 3 != 0:
        raise ValueError(f"pcm24 length must be multiple of 3, got {len(pcm24)}")

    out = bytearray((len(pcm24) // 3) * 4)
    j = 0
    for i in range(0, len(pcm24), 3):
        b0 = pcm24[i]
        b1 = pcm24[i + 1]
        b2 = pcm24[i + 2]
        out[j] = b0
        out[j + 1] = b1
        out[j + 2] = b2
        out[j + 3] = 0xFF if (b2 & 0x80) else 0x00
        j += 4
    return bytes(out)


class AudioOutputSink:
    """Near-real-time audio playback sink.

    Accepts PCM24LE (packed) bytes and plays through a RawOutputStream.

    The sink also emits an event every time playback transitions from silence
    (internal buffer empty) to producing non-empty output. We use this as a
    proxy for "first audio played" to compute TTFA.
    """

    def __init__(self, cfg: AudioOutputConfig):
        self._cfg = cfg
        self._stream: sd.RawOutputStream | None = None

        self._buf = bytearray()
        self._lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._play_epoch = 0
        self._awaiting_play_epoch: int | None = None
        self._play_started_q: asyncio.Queue[int] = asyncio.Queue()

    def start(self, *, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

        def _callback(outdata: bytearray, frames: int, time_info, status: sd.CallbackFlags) -> None:  # noqa: ANN001
            if status:
                pass

            want = frames * self._cfg.channels * 4  # int32 bytes
            got = 0
            with self._lock:
                if self._buf:
                    take = min(want, len(self._buf))
                    outdata[:take] = self._buf[:take]
                    del self._buf[:take]
                    got = take

            if got < want:
                outdata[got:want] = b"\x00" * (want - got)

            # If we actually output anything from the internal buffer and we're
            # waiting for a play-start marker, emit it.
            if got > 0 and self._awaiting_play_epoch is not None and self._loop is not None:
                epoch = self._awaiting_play_epoch
                self._awaiting_play_epoch = None

                def _emit() -> None:
                    try:
                        self._play_started_q.put_nowait(epoch)
                    except Exception:  # noqa: BLE001
                        pass

                self._loop.call_soon_threadsafe(_emit)

        self._stream = sd.RawOutputStream(
            device=self._cfg.device,
            samplerate=self._cfg.sample_rate,
            channels=self._cfg.channels,
            dtype="int32",
            callback=_callback,
        )
        self._stream.start()

        logger.info(
            "audio_out_started",
            extra={
                "device": self._cfg.device,
                "sample_rate": self._cfg.sample_rate,
                "channels": self._cfg.channels,
                "dtype": "int32",
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
        logger.info("audio_out_stopped")

    async def __aenter__(self) -> "AudioOutputSink":
        loop = asyncio.get_running_loop()
        self.start(loop=loop)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    def append_pcm24(self, pcm24: bytes) -> None:
        pcm32 = _pcm24le_to_int32le(pcm24)
        with self._lock:
            was_empty = not self._buf
            self._buf.extend(pcm32)
            if was_empty and self._awaiting_play_epoch is None:
                self._play_epoch += 1
                self._awaiting_play_epoch = self._play_epoch

    async def next_play_started(self) -> int:
        """Wait for the next playback-start marker."""

        return await self._play_started_q.get()


def decode_base64_audio(b64: str) -> bytes:
    return base64.b64decode(b64)
