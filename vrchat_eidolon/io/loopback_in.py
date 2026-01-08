from __future__ import annotations

import asyncio
import csv
import io
import logging
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from proctap import ProcessAudioCapture, STANDARD_CHANNELS, STANDARD_SAMPLE_RATE


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessLoopbackInputConfig:
    """Configuration for process loopback audio capture (Windows 20H1+)."""

    pid: int | None = None
    process_name: str | None = None

    # Target chunk size exposed to the rest of the app.
    chunk_ms: int = 100

    # Backpressure: keep it bounded and drop oldest chunks.
    queue_max_chunks: int = 20


def _require_windows_20h1_plus() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Process loopback capture is only supported on Windows")

    # Windows 10 20H1 is build 19041.
    try:
        build = sys.getwindowsversion().build
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to detect Windows build: {e}") from e

    if build < 19041:
        raise RuntimeError(
            f"Process loopback requires Windows 10 20H1+ (build 19041+), detected build={build}"
        )


def _resolve_pid(*, pid: int | None, process_name: str | None) -> int:
    if pid is not None:
        return int(pid)

    # Default target for local VRChat closed-loop MVP.
    name = (process_name or "VRChat.exe").strip()
    if not name:
        name = "VRChat.exe"

    # Use tasklist to avoid extra dependencies (e.g. psutil).
    # Example CSV row: "VRChat.exe","1234","Console","1","200,000 K"
    out = subprocess.check_output(
        ["tasklist", "/fi", f"imagename eq {name}", "/fo", "csv", "/nh"],
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    rows = list(csv.reader(io.StringIO(out)))
    matches: list[int] = []
    for row in rows:
        if len(row) < 2:
            continue
        image, pid_str = row[0], row[1]
        if image.lower() != name.lower():
            continue
        try:
            matches.append(int(pid_str))
        except ValueError:
            continue

    if not matches:
        raise RuntimeError(
            f"Process not found: {name}. "
            f"Start the process first, or set audio.loopback.pid / --loopback-pid."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple processes matched {name}: pids={matches}. "
            f"Set audio.loopback.pid / --loopback-pid to disambiguate."
        )

    return matches[0]


def float32le_to_pcm16le(raw: bytes, *, channels: int) -> tuple[bytes, bytes]:
    """Convert float32LE interleaved audio to PCM16LE.

    proctap returns audio in float32, normalized to [-1.0, 1.0]. We convert to
    int16 and preserve the channel layout.

    Returns:
        (pcm16_bytes, tail_bytes)
        - pcm16_bytes: converted, frame-aligned bytes
        - tail_bytes: leftover raw bytes that were not frame-aligned
    """

    if not raw:
        return b"", b""

    if channels not in (1, 2):
        raise ValueError(f"Unsupported channels={channels}; expected 1 or 2")

    frame_bytes_f32 = 4 * channels
    n = (len(raw) // frame_bytes_f32) * frame_bytes_f32
    if n <= 0:
        return b"", raw

    head = raw[:n]
    tail = raw[n:]

    # '<f4' ensures little-endian float32.
    arr = np.frombuffer(head, dtype="<f4")
    arr = np.clip(arr, -1.0, 1.0)

    # Scale into int16 range.
    i16 = (arr * 32767.0).astype("<i2")
    return i16.tobytes(), tail


class ProcessLoopbackInput:
    """Process loopback audio capture using proctap.

    Produces PCM16LE bytes compatible with the rest of the Realtime pipeline.

    Notes:
    - proctap outputs float32 stereo @ 48kHz by default.
    - We keep an internal bounded queue and drop oldest chunks on overflow.
    """

    def __init__(self, cfg: ProcessLoopbackInputConfig):
        self._cfg = cfg
        self._cap: ProcessAudioCapture | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._q: asyncio.Queue[bytes] | None = None

        self._raw_tail = bytearray()

    @property
    def sample_rate(self) -> int:
        return int(STANDARD_SAMPLE_RATE)

    @property
    def channels(self) -> int:
        return int(STANDARD_CHANNELS)

    async def __aenter__(self) -> "ProcessLoopbackInput":
        _require_windows_20h1_plus()

        pid = _resolve_pid(pid=self._cfg.pid, process_name=self._cfg.process_name)
        logger.info("loopback_in_starting", extra={"pid": pid, "chunk_ms": self._cfg.chunk_ms})

        self._cap = ProcessAudioCapture(pid)
        self._cap.start()

        self._q = asyncio.Queue(maxsize=self._cfg.queue_max_chunks)
        self._pump_task = asyncio.create_task(self._pump(), name="loopback_in_pump")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except Exception:  # noqa: BLE001
                pass
            self._pump_task = None

        if self._cap is not None:
            try:
                self._cap.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._cap.close()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

        self._q = None
        self._raw_tail.clear()
        logger.info("loopback_in_stopped")

    async def _pump(self) -> None:
        assert self._cap is not None
        assert self._q is not None

        frames_per_chunk = int(self.sample_rate * self._cfg.chunk_ms / 1000)
        out_frame_bytes = 2 * self.channels
        target_bytes = frames_per_chunk * out_frame_bytes

        buf = bytearray()

        async for raw in self._cap.iter_chunks():
            if not raw:
                continue

            self._raw_tail.extend(raw)
            pcm16, tail = float32le_to_pcm16le(bytes(self._raw_tail), channels=self.channels)
            self._raw_tail = bytearray(tail)

            if pcm16:
                buf.extend(pcm16)

            while len(buf) >= target_bytes:
                chunk = bytes(buf[:target_bytes])
                del buf[:target_bytes]

                try:
                    self._q.put_nowait(chunk)
                except asyncio.QueueFull:
                    # Drop oldest then retry.
                    try:
                        _ = self._q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._q.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # Give up if still full.
                        pass

    async def get_chunk(self, *, timeout_s: float | None) -> bytes | None:
        assert self._q is not None

        if timeout_s is None:
            return await self._q.get()

        try:
            return await asyncio.wait_for(self._q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def chunks(self):  # noqa: ANN201
        while True:
            c = await self.get_chunk(timeout_s=None)
            if c is None:
                continue
            yield c
