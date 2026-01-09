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
from vrchat_eidolon.io.rate_convert import PcmRateConverter


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QwenRealtimeConfig:
    url: str
    model: str
    voice: str = "Cherry"
    instructions: str = (
        "你在 VRChat 聊天：回复尽量简短（1-3 句）。用猫娘口吻，俏皮但不油腻。"
        "默认在整条回复最后加一个“喵”（除非用户明确要求不要）。"
    )
    turn_threshold: float = 0.5
    silence_duration_ms: int = 500
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm24"

    # Wire-format rates are not configurable in session.update. The official
    # DashScope samples for Omni Realtime use 16 kHz input and 24 kHz output.
    # We therefore resample locally when device rates differ.
    input_sample_rate_hz: int = 16000
    output_sample_rate_hz: int = 24000

    # Qwen Realtime audio is mono in the reference samples.
    input_channels: int = 1
    output_channels: int = 1
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

        # Map output "play epochs" to turn ids, so we can attribute the first
        # audible output to the correct turn even under cancellation.
        epoch_to_turn: dict[int, str] = {}

        # Track current response lifecycle for cancel behavior.
        active_response_id: str | None = None
        cancelled_response_ids: set[str] = set()
        last_cancel_ms: int = 0

        # Best-effort audio mapping to avoid the classic "garbled noise" symptom
        # when device sample rates don't match the model wire format.
        in_converter: PcmRateConverter | None = None
        if audio_in.sample_rate != self._cfg.input_sample_rate_hz or audio_in.channels != self._cfg.input_channels:
            in_converter = PcmRateConverter(
                sample_width_bytes=2,
                in_channels=audio_in.channels,
                in_sample_rate_hz=audio_in.sample_rate,
                out_channels=self._cfg.input_channels,
                out_sample_rate_hz=self._cfg.input_sample_rate_hz,
            )
            logger.warning(
                "audio_in_adapt",
                extra={
                    "device_rate_hz": audio_in.sample_rate,
                    "device_channels": audio_in.channels,
                    "wire_rate_hz": self._cfg.input_sample_rate_hz,
                    "wire_channels": self._cfg.input_channels,
                },
            )

        out_converter: PcmRateConverter | None = None
        if audio_out.sample_rate != self._cfg.output_sample_rate_hz or audio_out.channels != self._cfg.output_channels:
            out_converter = PcmRateConverter(
                # We down-convert to PCM16LE locally for stable playback.
                sample_width_bytes=2,
                in_channels=self._cfg.output_channels,
                in_sample_rate_hz=self._cfg.output_sample_rate_hz,
                out_channels=audio_out.channels,
                out_sample_rate_hz=audio_out.sample_rate,
            )
            logger.warning(
                "audio_out_adapt",
                extra={
                    "wire_rate_hz": self._cfg.output_sample_rate_hz,
                    "wire_channels": self._cfg.output_channels,
                    "device_rate_hz": audio_out.sample_rate,
                    "device_channels": audio_out.channels,
                },
            )

        async def _play_tracker() -> None:
            while True:
                epoch = await audio_out.next_play_started()
                turn_id = epoch_to_turn.pop(epoch, None)
                if turn_id is None:
                    continue

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

        async with websockets.connect(url, additional_headers=headers, ping_interval=20, ping_timeout=20) as ws:
            logger.info("realtime_connected", extra={"url": self._cfg.url, "model": self._cfg.model})

            send_lock = asyncio.Lock()

            async def _send(payload: Mapping[str, Any]) -> None:
                # websockets.send() is not safe to call concurrently.
                async with send_lock:
                    await ws.send(json.dumps(payload, ensure_ascii=False))

            await _send(
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
                }
            )

            def _should_barge_in_cancel() -> bool:
                # Best-effort: treat "audible recently" or "buffered" as speaking.
                return audio_out.is_audible(within_ms=400) or audio_out.pending_bytes() > 0

            async def _cancel_active_response(*, reason: str) -> None:
                nonlocal active_response_id, last_cancel_ms

                now = monotonic_ms()
                if now - last_cancel_ms < 400:
                    return
                last_cancel_ms = now

                if active_response_id is not None:
                    cancelled_response_ids.add(active_response_id)

                    # Request server-side cancellation (VAD mode can have an in-flight response).
                    # Spec: client event type=response.cancel
                    try:
                        await _send({"event_id": _event_id(), "type": "response.cancel"})
                    except Exception as e:  # noqa: BLE001
                        logger.warning("response_cancel_send_failed", extra={"error": str(e), "reason": reason})

                    # Debounce follow-up cancels; response.done will arrive.
                    active_response_id = None

                dropped = audio_out.flush()
                epoch_to_turn.clear()
                logger.info(
                    "barge_in_cancel",
                    extra={
                        "reason": reason,
                        "active_response_id": active_response_id,
                        "dropped_bytes": dropped,
                    },
                )

            async def _sender() -> None:
                # Avoid blocking forever when silent so we can react promptly to
                # websocket disconnects and session rotation.
                while True:
                    chunk = await audio_in.get_chunk(timeout_s=0.2)
                    if chunk is None:
                        continue

                    send_chunk = chunk
                    if in_converter is not None:
                        send_chunk = in_converter.convert(send_chunk)

                    b64 = base64.b64encode(send_chunk).decode("ascii")
                    await _send(
                        {
                            "event_id": _event_id(),
                            "type": "input_audio_buffer.append",
                            "audio": b64,
                        }
                    )

            async def _rotate_session_timer() -> None:
                # DashScope closes sessions at ~30 minutes; rotate a bit earlier.
                await asyncio.sleep(self._cfg.session_max_age_s)
                logger.info("realtime_session_rotation_requested", extra={"max_age_s": self._cfg.session_max_age_s})
                try:
                    await ws.close(code=1000, reason="session rotation")
                except Exception:  # noqa: BLE001
                    pass

            async def _receiver() -> None:
                # The model streams base64-encoded PCM bytes. Deltas may split
                # arbitrarily, so we must preserve sample alignment.
                #
                # NOTE:
                # DashScope's WebSocket protocol exposes output_audio_format as
                # an enum value (e.g. "pcm24" for qwen3-omni-flash-realtime).
                # The DashScope Python SDK documents this as
                # PCM_24000HZ_MONO_16BIT, i.e. 24 kHz sample rate with 16-bit
                # little-endian PCM samples. Treating this as 24-bit audio will
                # cause classic "high pitch + loud noise" corruption.
                pcm_tail = bytearray()
                bytes_per_sample = 2
                misaligned_chunks = 0

                async for msg in ws:
                    data = json.loads(msg)
                    typ = data.get("type")

                    if typ == "error":
                        logger.error("realtime_error", extra={"error": data.get("error")})
                        continue

                    if typ in {"session.created", "session.updated"}:
                        logger.info("realtime_session", extra={"type": typ, "session": data.get("session")})
                        continue

                    if typ == "response.created":
                        resp = data.get("response")
                        resp_id = None
                        if isinstance(resp, dict):
                            resp_id = resp.get("id")
                        if isinstance(resp_id, str):
                            active_response_id = resp_id
                            logger.info("response_created", extra={"response_id": resp_id})
                        continue

                    if typ == "response.done":
                        resp = data.get("response")
                        resp_id = None
                        if isinstance(resp, dict):
                            resp_id = resp.get("id")
                        if isinstance(resp_id, str):
                            logger.info("response_done", extra={"response_id": resp_id})
                            if active_response_id == resp_id:
                                active_response_id = None
                        continue

                    if typ == "input_audio_buffer.speech_started":
                        # Barge-in: user starts speaking while assistant is speaking.
                        if _should_barge_in_cancel():
                            await _cancel_active_response(reason="speech_started")
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
                        response_id = data.get("response_id")
                        if not isinstance(delta, str):
                            continue

                        if isinstance(response_id, str) and response_id in cancelled_response_ids:
                            continue

                        raw = base64.b64decode(delta)
                        if len(raw) % bytes_per_sample != 0:
                            # Keep it as a warning (and rate-limit) to catch
                            # wire-format mismatches without spamming logs.
                            misaligned_chunks += 1
                            if misaligned_chunks <= 3:
                                logger.warning(
                                    "audio_wire_chunk_not_sample_aligned",
                                    extra={"len": len(raw), "bytes_per_sample": bytes_per_sample},
                                )

                        pcm_tail.extend(raw)
                        if not pcm_tail:
                            continue

                        frame_bytes_wire = bytes_per_sample * self._cfg.output_channels

                        # Only process full frames to avoid byte misalignment.
                        n = (len(pcm_tail) // frame_bytes_wire) * frame_bytes_wire
                        if n <= 0:
                            continue

                        pcm = bytes(pcm_tail[:n])
                        del pcm_tail[:n]

                        # Wire format is assumed PCM16LE at cfg.output_sample_rate_hz.
                        pcm16 = pcm
                        if out_converter is not None:
                            pcm16 = out_converter.convert(pcm16)

                        epoch = audio_out.append_pcm16(pcm16)

                        if isinstance(item_id, str):
                            t = turns.get(item_id)
                            if t is None:
                                t = TurnTtfa(turn_id=item_id)
                                turns[item_id] = t
                            if t.first_audio_delta_ms is None:
                                t.first_audio_delta_ms = monotonic_ms()
                                if epoch is not None:
                                    epoch_to_turn[epoch] = item_id
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

                # Exiting the receive loop means the websocket closed.
                # Raise to force TaskGroup cancellation even if sender is idle.
                raise ConnectionError("websocket receive loop ended")

            async with asyncio.TaskGroup() as tg:
                tg.create_task(_play_tracker())
                tg.create_task(_sender())
                tg.create_task(_receiver())
                tg.create_task(_rotate_session_timer())

            # If we ever get here, the session ended.
            logger.info("realtime_disconnected")

        age_s = (monotonic_ms() - session_started_ms) / 1000.0
        if age_s > self._cfg.session_max_age_s:
            logger.info("realtime_session_rotated", extra={"age_s": age_s})
