from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from vrchat_eidolon.io.audio_in import AudioInput, AudioInputConfig
from vrchat_eidolon.io.audio_out import AudioOutputConfig, AudioOutputSink
from vrchat_eidolon.llm.qwen_realtime import QwenRealtimeClient, QwenRealtimeConfig


logger = logging.getLogger(__name__)


def _get(d: Mapping[str, Any], path: str, default: Any) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


async def run_speech_loop(cfg: Mapping[str, Any]) -> None:
    """Run the Milestone 1 Speech Loop (Realtime).

    This is intentionally independent from LangGraph (turn-level orchestration).
    """

    api_key = _get(cfg, "qwen.api_key", None)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Missing qwen.api_key (expected a non-empty string)")

    url = _get(cfg, "qwen.realtime.url", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
    model = _get(cfg, "qwen.realtime.model", "qwen3-omni-flash-realtime")
    voice = _get(cfg, "qwen.realtime.voice", "Cherry")
    instructions = _get(cfg, "qwen.realtime.instructions", "You are a helpful assistant.")
    turn_threshold = float(_get(cfg, "qwen.realtime.turn_detection.threshold", 0.5))

    sample_rate_in = int(_get(cfg, "audio.input.sample_rate", 48000))
    channels_in = int(_get(cfg, "audio.input.channels", 1))
    device_in = _get(cfg, "audio.input.device", None)

    sample_rate_out = int(_get(cfg, "audio.output.sample_rate", 48000))
    channels_out = int(_get(cfg, "audio.output.channels", 1))
    device_out = _get(cfg, "audio.output.device", None)

    chunk_ms = int(_get(cfg, "audio.input.chunk_ms", 100))
    silence_duration_ms = int(_get(cfg, "audio.vad.silence_duration_ms", 500))

    logger.info(
        "speech_loop_config",
        extra={
            "ws_url": url,
            "model": model,
            "voice": voice,
            "chunk_ms": chunk_ms,
            "silence_duration_ms": silence_duration_ms,
        },
    )

    ai_cfg = QwenRealtimeConfig(
        url=str(url),
        model=str(model),
        voice=str(voice),
        instructions=str(instructions),
        turn_threshold=turn_threshold,
        silence_duration_ms=silence_duration_ms,
    )

    in_cfg = AudioInputConfig(
        device=device_in,
        sample_rate=sample_rate_in,
        channels=channels_in,
        chunk_ms=chunk_ms,
    )

    out_cfg = AudioOutputConfig(
        device=device_out,
        sample_rate=sample_rate_out,
        channels=channels_out,
    )

    client = QwenRealtimeClient(cfg=ai_cfg, api_key=api_key)

    async with AudioInput(in_cfg) as audio_in, AudioOutputSink(out_cfg) as audio_out:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(client.run(audio_in=audio_in, audio_out=audio_out))
