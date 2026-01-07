from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import websockets

from vrchat_eidolon.core.clock import TurnTtfa, monotonic_ms
from vrchat_eidolon.io.audio_in import AudioInput
from vrchat_eidolon.io.audio_out import AudioOutputSink


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QwenRealtimeConfig:
    url: str
    model: str
    voice: str = "Cherry"
    instructions: str = "You are a helpful assistant."
    turn_threshold: float = 0.5
    silence_duration_ms: int = 500
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm24"
    session_max_age_s: int = 28 * 60


def _event_id() -> str:
    return f"event_{monotonic_ms()}_{random.randint(1000, 9999)}"


class QwenRealtimeClient:
    """Minimal Qwen-Omni-Realtime WebSocket client (VAD mode)."""

    def __init__(self, *, cfg: QwenRealtimeConfig, api_key: str):
        self._cfg = cfg
        self._api_key = api_key

    async def run(self, *, audio_in: AudioInput, audio_out: AudioOutputSink) -> None:
        backoff_s = 0.5
        while True:
            try:
                await self._run_one_session(audio_in=audio_in, audio_out=audio_out)
                backoff_s = 0.5
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "realtime_session_error",
                    extra={"error": str(e), "backoff_s": backoff_s},
                )
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 10.0)

    async def _run_one_session(self, *, audio_in: AudioInput, audio_out: AudioOutputSink) -> None:
        url = f"{self._cfg.url}?model={self._cfg.model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        session_started_ms = monotonic_ms()

        # Track TTFA per item_id (turn).
        turns: dict[str, TurnTtfa] = {}
        pending_play_turns: deque[str] = deque()

        async def _play_tracker() -> None:
            while True:
                _ = await audio_out.next_play_started()
                # Attribute play-start to the earliest turn that has audio but
                # hasn't yet been marked as played.
                while pending_play_turns:
                    turn_id = pending_play_turns.popleft()
                    t = turns.get(turn_id)
                    if t is None or t.first_audio_played_ms is not None:
                        continue
                    t.first_audio_played_ms = monotonic_ms()
                    logger.info(
                        "ttfa",
                        extra={
                            "turn_id": turn_id,
                            "eos_proxy_ms": t.eos_proxy_ms,
                            "first_audio_delta_ms": t.first_audio_delta_ms,
                            "first_audio_played_ms": t.first_audio_played_ms,
                            "ttf_delta_ms": t.ttf_delta_ms(),
                            "ttfa_ms": t.ttfa_ms(),
                        },
                    )
                    break

        async with websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20) as ws:
            logger.info("realtime_connected", extra={"url": self._cfg.url, "model": self._cfg.model})

            await ws.send(
                json.dumps(
                    {
                        "event_id": _event_id(),
                        "type": "session.update",
                        "session": {
                            "modalities": ["text", "audio"],
                            "voice": self._cfg.voice,
                            "input_audio_format": self._cfg.input_audio_format,
                            "output_audio_format": self._cfg.output_audio_format,
                            "instructions": self._cfg.instructions,
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": self._cfg.turn_threshold,
                                "silence_duration_ms": self._cfg.silence_duration_ms,
                            },
                        },
                    },
                    ensure_ascii=False,
                )
            )

            async def _sender() -> None:
                async for chunk in audio_in.chunks():
                    b64 = base64.b64encode(chunk).decode("ascii")
                    await ws.send(
                        json.dumps(
                            {
                                "event_id": _event_id(),
                                "type": "input_audio_buffer.append",
                                "audio": b64,
                            }
                        )
                    )

            async def _receiver() -> None:
                async for msg in ws:
                    data = json.loads(msg)
                    typ = data.get("type")

                    if typ == "error":
                        logger.error("realtime_error", extra={"error": data.get("error")})
                        continue

                    if typ in {"session.created", "session.updated"}:
                        logger.info("realtime_session", extra={"type": typ, "session": data.get("session")})
                        continue

                    if typ == "input_audio_buffer.speech_stopped":
                        item_id = data.get("item_id")
                        if isinstance(item_id, str):
                            turns[item_id] = TurnTtfa(turn_id=item_id, eos_proxy_ms=monotonic_ms())
                            logger.info(
                                "speech_stopped",
                                extra={"turn_id": item_id, "audio_end_ms": data.get("audio_end_ms")},
                            )
                        continue

                    if typ == "conversation.item.input_audio_transcription.completed":
                        item_id = data.get("item_id")
                        transcript = data.get("transcript")
                        logger.info(
                            "asr_completed",
                            extra={"turn_id": item_id, "transcript": transcript},
                        )
                        continue

                    if typ == "response.audio_transcript.delta":
                        logger.debug(
                            "tts_transcript_delta",
                            extra={"delta": data.get("delta"), "response_id": data.get("response_id")},
                        )
                        continue

                    if typ == "response.audio.delta":
                        delta = data.get("delta")
                        item_id = data.get("item_id")
                        if not isinstance(delta, str):
                            continue

                        pcm24 = base64.b64decode(delta)
                        audio_out.append_pcm24(pcm24)

                        if isinstance(item_id, str):
                            t = turns.get(item_id)
                            if t is None:
                                t = TurnTtfa(turn_id=item_id)
                                turns[item_id] = t
                            if t.first_audio_delta_ms is None:
                                t.first_audio_delta_ms = monotonic_ms()
                                pending_play_turns.append(item_id)
                                logger.info(
                                    "first_audio_delta",
                                    extra={
                                        "turn_id": item_id,
                                        "eos_proxy_ms": t.eos_proxy_ms,
                                        "first_audio_delta_ms": t.first_audio_delta_ms,
                                    },
                                )
                        continue

                    if typ == "response.audio.done":
                        logger.info(
                            "audio_done",
                            extra={"response_id": data.get("response_id"), "item_id": data.get("item_id")},
                        )
                        continue

                    # Keep other events at debug; they can be noisy.
                    logger.debug("realtime_event", extra={"type": typ, "data": data})

            async with asyncio.TaskGroup() as tg:
                tg.create_task(_play_tracker())
                tg.create_task(_sender())
                tg.create_task(_receiver())

            # If we ever get here, the session ended.
            logger.info("realtime_disconnected")

        age_s = (monotonic_ms() - session_started_ms) / 1000.0
        if age_s > self._cfg.session_max_age_s:
            logger.info("realtime_session_rotated", extra={"age_s": age_s})
