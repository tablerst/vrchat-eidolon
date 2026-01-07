from __future__ import annotations

from array import array

import pytest

from vrchat_eidolon.io.rate_convert import PcmRateConverter


def _pcm16_ramp(samples: int) -> bytes:
    a = array("h", [i % 32768 for i in range(samples)])
    return a.tobytes()


def test_pcm16_rate_convert_length_scales_reasonably() -> None:
    # 100 ms @ 16 kHz mono int16
    src = _pcm16_ramp(1600)

    conv = PcmRateConverter(
        sample_width_bytes=2,
        in_channels=1,
        in_sample_rate_hz=16000,
        out_channels=1,
        out_sample_rate_hz=24000,
    )
    out = conv.convert(src)

    # Expect ~1.5x samples; allow some slack for internal filter state.
    expected = int(len(src) * 24000 / 16000)
    assert abs(len(out) - expected) < 64


def test_pcm24_rate_convert_accepts_data() -> None:
    # 10 ms @ 24 kHz mono pcm24 => 240 samples => 720 bytes
    src = bytes([0, 0, 0] * 240)

    conv = PcmRateConverter(
        sample_width_bytes=3,
        in_channels=1,
        in_sample_rate_hz=24000,
        out_channels=1,
        out_sample_rate_hz=96000,
    )

    out = conv.convert(src)
    assert len(out) > len(src)
    assert len(out) % 3 == 0


def test_channel_mapping_rejects_unsupported() -> None:
    conv = PcmRateConverter(
        sample_width_bytes=2,
        in_channels=3,
        in_sample_rate_hz=16000,
        out_channels=1,
        out_sample_rate_hz=16000,
    )
    with pytest.raises(ValueError):
        _ = conv.convert(b"\x00" * 6)
