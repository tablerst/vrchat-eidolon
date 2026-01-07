from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import audioop

import sounddevice as sd


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioOutputConfig:
    device: str | int | None
    sample_rate: int
    channels: int


class AudioOutputSink:
    """Near-real-time audio playback sink.

    Accepts PCM16LE bytes for playback through a RawOutputStream.

    We also provide a convenience method to accept PCM24LE (packed) and down-convert
    to PCM16LE for playback stability.

    The sink also emits an event every time playback transitions from silence
    (internal buffer empty) to producing non-empty output. We use this as a
    proxy for "first audio played" to compute TTFA.
    """

    def __init__(self, cfg: AudioOutputConfig):
        self._cfg = cfg
        self._stream: sd.RawOutputStream | None = None

        self._buf = bytearray()
        self._tail = bytearray()
        self._lock = threading.Lock()

        self._effective_sample_rate: int | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._play_epoch = 0
        self._awaiting_play_epoch: int | None = None
        self._play_started_q: asyncio.Queue[int] = asyncio.Queue()

    @property
    def device(self) -> str | int | None:
        return self._cfg.device

    @property
    def sample_rate(self) -> int:
        return self._effective_sample_rate or self._cfg.sample_rate

    @property
    def channels(self) -> int:
        return self._cfg.channels

    def start(self, *, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

        def _callback(outdata: bytearray, frames: int, time_info, status: sd.CallbackFlags) -> None:  # noqa: ANN001
            if status:
                pass

            # For RawOutputStream, `outdata` is a byte buffer sized to
            # frames * channels * bytes_per_sample.
            want = len(outdata)  # PCM16LE bytes
            got = 0
            with self._lock:
                if self._buf:
                    # Only output whole frames (avoid half-sample artifacts).
                    frame_bytes = 2 * self._cfg.channels
                    take = min(want, len(self._buf))
                    take = (take // frame_bytes) * frame_bytes
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
            dtype="int24",
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
            "audio_out_started",
            extra={
                "device": self._cfg.device,
                "sample_rate": self._cfg.sample_rate,
                "effective_sample_rate": self.sample_rate,
                "channels": self._cfg.channels,
                "dtype": "int24",
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
        self._effective_sample_rate = None

    async def __aenter__(self) -> "AudioOutputSink":
        loop = asyncio.get_running_loop()
        self.start(loop=loop)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()

    def append_pcm16(self, pcm16: bytes) -> None:
        if not pcm16:
            return

        frame_bytes = 2 * self._cfg.channels
        with self._lock:
            was_empty = not self._buf

            # Keep internal buffer frame-aligned.
            self._tail.extend(pcm16)
            n = (len(self._tail) // frame_bytes) * frame_bytes
            if n:
                self._buf.extend(self._tail[:n])
                del self._tail[:n]

            if was_empty and self._buf and self._awaiting_play_epoch is None:
                self._play_epoch += 1
                self._awaiting_play_epoch = self._play_epoch

    def append_pcm24(self, pcm24: bytes) -> None:
        """Append PCM24LE (packed) by downconverting to PCM16LE."""

        if len(pcm24) % 3 != 0:
            raise ValueError(f"pcm24 length must be multiple of 3, got {len(pcm24)}")

        # Down-convert for broader device compatibility.
        pcm16 = audioop.lin2lin(pcm24, 3, 2)
        self.append_pcm16(pcm16)

    async def next_play_started(self) -> int:
        """Wait for the next playback-start marker."""

        return await self._play_started_q.get()


