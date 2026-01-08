from __future__ import annotations

import numpy as np

from vrchat_eidolon.io.loopback_in import float32le_to_pcm16le


def test_float32le_to_pcm16le_mono_clip_and_tail() -> None:
    samples = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    raw = samples.tobytes() + b"\x99"  # force non-frame-aligned tail

    pcm16, tail = float32le_to_pcm16le(raw, channels=1)
    assert tail == b"\x99"

    out = np.frombuffer(pcm16, dtype="<i2")
    assert out.tolist() == [0, 32767, -32767, 32767, -32767]


def test_float32le_to_pcm16le_stereo_layout_preserved() -> None:
    # Two stereo frames: (L0,R0,L1,R1)
    samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    pcm16, tail = float32le_to_pcm16le(samples.tobytes(), channels=2)
    assert tail == b""

    out = np.frombuffer(pcm16, dtype="<i2")
    # 0.5 -> 16383.5 -> truncates toward zero in astype
    assert out.tolist() == [0, 16383, -16383, 32767]
