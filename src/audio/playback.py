from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from observability.logging import get_logger


class AudioPlayback:
    def start(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def push_audio_b64(self, chunk_b64: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def play_audio_chunks(self, chunks_b64: Iterable[str]) -> None:
        """Compatibility helper: play an iterable of base64 chunks."""

        self.start()
        try:
            for b64 in chunks_b64:
                self.push_audio_b64(str(b64))
        finally:
            self.close()


@dataclass(slots=True)
class _WavFormat:
    channels: int
    sample_rate: int
    bits_per_sample: int



class NullAudioPlayback(AudioPlayback):
    """No-op playback: consumes the iterator but does not output sound."""

    def start(self) -> None:
        return

    def push_audio_b64(self, chunk_b64: str) -> None:
        _ = chunk_b64

    def close(self) -> None:
        return


class _WavStreamParser:
    """Incremental WAV parser that yields PCM bytes from the 'data' chunk.

    Limitations:
    - PCM (format=1) only
    - int16 only
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._fmt: _WavFormat | None = None
        self._data_remaining: int | None = None
        self._in_data = False
        self._riff_ok = False

    @property
    def fmt(self) -> _WavFormat | None:
        return self._fmt

    def feed(self, data: bytes) -> list[bytes]:
        if data:
            self._buf.extend(data)

        out: list[bytes] = []

        # Parse RIFF header.
        if not self._riff_ok:
            if len(self._buf) < 12:
                return out
            if self._buf[0:4] != b"RIFF" or self._buf[8:12] != b"WAVE":
                raise ValueError("not a RIFF/WAVE stream")
            del self._buf[:12]
            self._riff_ok = True

        # Parse chunks.
        while True:
            if self._in_data:
                if self._data_remaining is None:
                    if self._buf:
                        out.append(bytes(self._buf))
                        self._buf.clear()
                    return out

                if self._data_remaining <= 0:
                    self._in_data = False
                    self._data_remaining = None
                    continue

                if not self._buf:
                    return out

                take = min(len(self._buf), int(self._data_remaining))
                out.append(bytes(self._buf[:take]))
                del self._buf[:take]
                self._data_remaining -= take
                continue

            if len(self._buf) < 8:
                return out

            chunk_id = bytes(self._buf[0:4])
            chunk_size = struct.unpack("<I", self._buf[4:8])[0]
            if len(self._buf) < 8 + chunk_size:
                return out

            chunk_data = bytes(self._buf[8 : 8 + chunk_size])
            del self._buf[: 8 + chunk_size]

            # Chunks are word-aligned.
            if chunk_size % 2 == 1 and self._buf:
                del self._buf[:1]

            if chunk_id == b"fmt ":
                if len(chunk_data) < 16:
                    raise ValueError("invalid fmt chunk")
                audio_format, channels, sample_rate = struct.unpack("<HHI", chunk_data[0:8])
                bits_per_sample = struct.unpack("<H", chunk_data[14:16])[0]
                if audio_format != 1:
                    raise ValueError(f"unsupported wav format: {audio_format}")
                self._fmt = _WavFormat(
                    channels=int(channels),
                    sample_rate=int(sample_rate),
                    bits_per_sample=int(bits_per_sample),
                )
            elif chunk_id == b"data":
                self._in_data = True
                self._data_remaining = int(chunk_size)
                if chunk_data:
                    out.append(chunk_data)
                    self._data_remaining -= len(chunk_data)
            else:
                # Skip other chunks.
                continue


class SoundDeviceAudioPlayback(AudioPlayback):
    """Play streamed base64 WAV deltas to an output device using sounddevice."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        preferred_sample_rate: int | None = None,
        preferred_channels: int | None = None,
    ) -> None:
        self._device = device
        self._preferred_sr = preferred_sample_rate
        self._preferred_ch = preferred_channels

        self._log = get_logger("eidolon.audio")
        self._sd: Any | None = None
        self._stream: Any | None = None

        self._b64_buf = ""
        self._wav = _WavStreamParser()

    def start(self) -> None:
        if self._sd is not None:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'sounddevice'. Install it with: uv add sounddevice") from e

        self._sd = sd

    def _resolve_device(self) -> str | int | None:
        if self._device is None:
            return None
        if isinstance(self._device, int):
            return self._device

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
                max_out = int(d.get("max_output_channels", 0) or 0)
            except Exception:
                continue
            if max_out <= 0:
                continue
            if needle in name.lower():
                matches.append(idx)

        if not matches:
            raise ValueError(f"output device not found: {self._device!r}")
        return matches[0]

    def _ensure_stream(self, *, fmt: _WavFormat) -> None:
        if self._stream is not None:
            return

        if fmt.bits_per_sample != 16:
            raise ValueError(f"unsupported bits_per_sample: {fmt.bits_per_sample}")

        if self._preferred_sr is not None and int(self._preferred_sr) != int(fmt.sample_rate):
            self._log.warning(
                "audio_output_sample_rate_mismatch",
                preferred=int(self._preferred_sr),
                actual=int(fmt.sample_rate),
            )

        if self._preferred_ch is not None and int(self._preferred_ch) != int(fmt.channels):
            self._log.warning(
                "audio_output_channels_mismatch",
                preferred=int(self._preferred_ch),
                actual=int(fmt.channels),
            )

        sd = self._sd
        assert sd is not None

        self._stream = sd.RawOutputStream(
            samplerate=int(fmt.sample_rate),
            channels=int(fmt.channels),
            dtype="int16",
            device=self._resolve_device(),
        )
        self._stream.start()
        self._log.info(
            "audio_output_started",
            sample_rate=int(fmt.sample_rate),
            channels=int(fmt.channels),
            device=self._device,
        )

    def push_audio_b64(self, chunk_b64: str) -> None:
        if not chunk_b64:
            return
        self.start()

        # Incremental base64 decode: decode as much prefix as possible.
        self._b64_buf += chunk_b64
        dec_len = (len(self._b64_buf) // 4) * 4
        if dec_len <= 0:
            return

        prefix = self._b64_buf[:dec_len]
        self._b64_buf = self._b64_buf[dec_len:]

        try:
            data = base64.b64decode(prefix, validate=False)
        except Exception as e:
            self._log.warning("audio_b64_decode_failed", exc=type(e).__name__)
            return

        try:
            pcm_parts = self._wav.feed(data)
        except Exception as e:
            self._log.warning("audio_wav_parse_failed", exc=type(e).__name__, error=str(e))
            return

        fmt = self._wav.fmt
        if fmt is None:
            return

        self._ensure_stream(fmt=fmt)

        for pcm in pcm_parts:
            if not pcm:
                continue
            try:
                self._stream.write(pcm)
            except Exception as e:
                self._log.warning("audio_output_write_failed", exc=type(e).__name__, error=str(e))

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                try:
                    self._stream.close()
                finally:
                    self._stream = None
