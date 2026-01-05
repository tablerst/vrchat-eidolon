from __future__ import annotations

from typing import Iterable


class AudioPlayback:
    def play_audio_chunks(self, chunks_b64: Iterable[str]) -> None:  # pragma: no cover
        raise NotImplementedError


class NullAudioPlayback(AudioPlayback):
    """No-op playback: consumes the iterator but does not output sound."""

    def play_audio_chunks(self, chunks_b64: Iterable[str]) -> None:
        for _ in chunks_b64:
            pass
