from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping

from vrchat_eidolon.io.audio_in import AudioInput, AudioInputConfig
from vrchat_eidolon.io.audio_out import AudioOutputConfig, AudioOutputSink
from vrchat_eidolon.io.loopback_in import ProcessLoopbackInput, ProcessLoopbackInputConfig
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

    wire_in_rate = int(_get(cfg, "qwen.realtime.input_sample_rate_hz", 16000))
    wire_out_rate = int(_get(cfg, "qwen.realtime.output_sample_rate_hz", 24000))
    wire_in_channels = int(_get(cfg, "qwen.realtime.input_channels", 1))
    wire_out_channels = int(_get(cfg, "qwen.realtime.output_channels", 1))

    sample_rate_in = int(_get(cfg, "audio.input.sample_rate", 48000))
    channels_in = int(_get(cfg, "audio.input.channels", 1))
    device_in = _get(cfg, "audio.input.device", None)
    input_source = str(_get(cfg, "audio.input.source", "mic"))

    loopback_pid = _get(cfg, "audio.loopback.pid", None)
    loopback_process_name = _get(cfg, "audio.loopback.process_name", "VRChat.exe")

    sample_rate_out = int(_get(cfg, "audio.output.sample_rate", 48000))
    channels_out = int(_get(cfg, "audio.output.channels", 1))
    device_out = _get(cfg, "audio.output.device", None)

    chunk_ms = int(_get(cfg, "audio.input.chunk_ms", 100))
    silence_duration_ms = int(_get(cfg, "audio.vad.silence_duration_ms", 500))

    logger.info(
        "speech_loop_config",
        extra={
            "input_source": input_source,
            "ws_url": url,
            "model": model,
            "voice": voice,
            "chunk_ms": chunk_ms,
            "silence_duration_ms": silence_duration_ms,
            "wire_in_rate_hz": wire_in_rate,
            "wire_out_rate_hz": wire_out_rate,
            "device_in_rate_hz": sample_rate_in,
            "device_out_rate_hz": sample_rate_out,
        },
    )

    ai_cfg = QwenRealtimeConfig(
        url=str(url),
        model=str(model),
        voice=str(voice),
        instructions=str(instructions),
        turn_threshold=turn_threshold,
        silence_duration_ms=silence_duration_ms,
        input_sample_rate_hz=wire_in_rate,
        output_sample_rate_hz=wire_out_rate,
        input_channels=wire_in_channels,
        output_channels=wire_out_channels,
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

    if input_source == "mic":
        audio_in_cm = AudioInput(in_cfg)
    elif input_source in {"process_loopback", "loopback"}:
        pid_val: int | None = None
        if isinstance(loopback_pid, int):
            pid_val = int(loopback_pid)
        elif isinstance(loopback_pid, str) and loopback_pid.strip().isdigit():
            pid_val = int(loopback_pid.strip())

        audio_in_cm = ProcessLoopbackInput(
            ProcessLoopbackInputConfig(
                pid=pid_val,
                process_name=str(loopback_process_name) if loopback_process_name is not None else None,
                chunk_ms=chunk_ms,
            )
        )
    else:
        raise ValueError(f"Unknown audio.input.source={input_source!r}; expected 'mic' or 'process_loopback'")

    async with audio_in_cm as audio_in, AudioOutputSink(out_cfg) as audio_out:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(client.run(audio_in=audio_in, audio_out=audio_out))
