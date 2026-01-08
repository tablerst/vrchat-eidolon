from __future__ import annotations

from vrchat_eidolon.io.audio_out import AudioOutputConfig, AudioOutputSink


def test_audio_out_epoch_and_flush() -> None:
    sink = AudioOutputSink(
        AudioOutputConfig(
            device=None,
            sample_rate=48_000,
            channels=1,
        )
    )

    # Two frames of PCM16LE mono (4 bytes).
    pcm = b"\x00\x00\x01\x00"

    epoch1 = sink.append_pcm16(pcm)
    assert isinstance(epoch1, int)
    assert sink.pending_bytes() == len(pcm)

    # Appending while buffered should not emit a new epoch.
    epoch_none = sink.append_pcm16(pcm)
    assert epoch_none is None
    assert sink.pending_bytes() == len(pcm) * 2

    dropped = sink.flush()
    assert dropped == len(pcm) * 2
    assert sink.pending_bytes() == 0

    epoch2 = sink.append_pcm16(pcm)
    assert isinstance(epoch2, int)
    assert epoch2 != epoch1
