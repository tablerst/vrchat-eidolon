from __future__ import annotations

from array import array

from audio.vad import EnergyVadSegmenter, VadParams


def _pcm_square_wave(*, frames: int, amp: float, channels: int = 1) -> bytes:
    amp = max(0.0, min(amp, 1.0))
    v = int(amp * 32767)

    a = array("h")
    for i in range(frames * channels):
        a.append(v if (i % 2) else -v)
    return a.tobytes()


def test_energy_vad_no_segment_on_silence() -> None:
    seg = EnergyVadSegmenter(
        sample_rate=48_000,
        channels=1,
        params=VadParams(silence_duration_ms=500, min_speech_duration_ms=300, energy_threshold=0.02),
    )

    frames = 480  # 10ms
    silence = _pcm_square_wave(frames=frames, amp=0.0)

    out = None
    for _ in range(200):  # 2s
        out = seg.push(pcm=silence, frames=frames) or out

    assert out is None


def test_energy_vad_segments_after_silence() -> None:
    seg = EnergyVadSegmenter(
        sample_rate=48_000,
        channels=1,
        params=VadParams(silence_duration_ms=500, min_speech_duration_ms=300, energy_threshold=0.02),
    )

    frames = 480  # 10ms
    speech = _pcm_square_wave(frames=frames, amp=0.10)
    silence = _pcm_square_wave(frames=frames, amp=0.0)

    out = None

    # 500ms speech
    for _ in range(50):
        out = seg.push(pcm=speech, frames=frames) or out
    assert out is None

    # 600ms silence should trigger end (after 500ms)
    for _ in range(60):
        out = seg.push(pcm=silence, frames=frames) or out
        if out is not None:
            break

    assert out is not None
    assert out.sample_rate == 48_000
    assert 480 <= out.duration_ms <= 520
    assert len(out.pcm) > 0


def test_energy_vad_discards_too_short_speech() -> None:
    seg = EnergyVadSegmenter(
        sample_rate=48_000,
        channels=1,
        params=VadParams(silence_duration_ms=200, min_speech_duration_ms=300, energy_threshold=0.02),
    )

    frames = 480  # 10ms
    speech = _pcm_square_wave(frames=frames, amp=0.10)
    silence = _pcm_square_wave(frames=frames, amp=0.0)

    out = None

    # 100ms speech (below min 300ms)
    for _ in range(10):
        out = seg.push(pcm=speech, frames=frames) or out
    assert out is None

    # silence triggers stop, but segment should be discarded
    for _ in range(40):
        out = seg.push(pcm=silence, frames=frames) or out

    assert out is None
