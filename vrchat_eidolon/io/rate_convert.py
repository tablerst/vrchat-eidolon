from __future__ import annotations

import audioop
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PcmRateConverter:
    """Streaming PCM channel + sample-rate converter.

    This uses the stdlib `audioop` module to avoid extra dependencies.

    Notes:
    - `audioop` is deprecated upstream, but it is still available on Python 3.12.
      If/when it disappears, we should replace this with a dedicated resampler.
    - Audio is assumed to be signed PCM in little-endian byte order.
    """

    sample_width_bytes: int
    in_channels: int
    in_sample_rate_hz: int
    out_channels: int
    out_sample_rate_hz: int

    _state: Any = None

    def convert(self, data: bytes) -> bytes:
        if not data:
            return b""

        if self.in_channels not in (1, 2):
            raise ValueError(f"Unsupported in_channels={self.in_channels}; expected 1 or 2")
        if self.out_channels not in (1, 2):
            raise ValueError(f"Unsupported out_channels={self.out_channels}; expected 1 or 2")

        w = self.sample_width_bytes

        # First convert channels (if needed) while keeping sample rate.
        if self.in_channels == 2 and self.out_channels == 1:
            data = audioop.tomono(data, w, 0.5, 0.5)
        elif self.in_channels == 1 and self.out_channels == 2:
            data = audioop.tostereo(data, w, 1.0, 1.0)
        elif self.in_channels != self.out_channels:
            raise ValueError(
                f"Unsupported channel mapping {self.in_channels}->{self.out_channels}; expected 1 or 2"
            )

        # Then convert sample rate.
        if self.in_sample_rate_hz == self.out_sample_rate_hz:
            return data

        out, self._state = audioop.ratecv(
            data,
            w,
            self.out_channels,
            self.in_sample_rate_hz,
            self.out_sample_rate_hz,
            self._state,
        )
        return out
