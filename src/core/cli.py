from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path

from audio import MicrophoneAudioCapture, SoundDeviceAudioPlayback, VadParams
from observability.logging import configure_logging, get_logger
from orchestrator.graph_orchestrator import GraphOrchestrator
from qwen.client import FakeQwenClient, QwenClient

from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VRChat Eidolon")
    p.add_argument("--config", default="configs/app.yaml", help="YAML 配置路径")
    p.add_argument("--log-level", default="INFO", help="日志级别")
    p.add_argument("--text", default="你好", help="MVP：文本输入")
    p.add_argument("--listen", action="store_true", help="MVP：麦克风 LISTEN（采集一段语音并退出）")
    p.add_argument("--listen-save", default=None, help="可选：把采集到的 utterance 保存为 WAV 文件")
    p.add_argument("--voice", action="store_true", help="MVP：语音回路（LISTEN -> PLAN(audio) -> SPEAK）")
    p.add_argument("--no-audio-out", action="store_true", help="语音回路：不播放音频，仅输出文本")
    p.add_argument("--fake", action="store_true", help="使用 FakeQwenClient（离线 stub）")
    return p


def _save_wav(*, pcm: bytes, path: str | Path, sample_rate: int, channels: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(int(channels))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    log = get_logger("eidolon.cli", level=args.log_level)

    # Offline stub: allow running without a real key.
    if args.fake and not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = "k_fake"

    cfg = load_config(args.config)

    vad = VadParams(
        silence_duration_ms=cfg.audio.vad.silence_duration_ms,
        min_speech_duration_ms=cfg.audio.vad.min_speech_duration_ms,
        energy_threshold=cfg.audio.vad.energy_threshold,
    )

    if args.voice:
        cap = MicrophoneAudioCapture(
            device=cfg.audio.input.device,
            sample_rate=cfg.audio.input.sample_rate,
            channels=cfg.audio.input.channels,
            vad=vad,
        )

        try:
            seg = cap.next_segment()
        finally:
            cap.close()

        log.info(
            "voice_listen_done",
            duration_ms=seg.duration_ms,
            sample_rate=seg.sample_rate,
            bytes=len(seg.pcm),
        )

        if args.fake:
            qwen = FakeQwenClient()
        else:
            qwen = QwenClient(cfg.qwen)

        mcp_servers = dict(cfg.mcp.servers) if cfg.mcp.enabled else None

        playback = (
            SoundDeviceAudioPlayback(
                device=cfg.audio.output.device,
                preferred_sample_rate=cfg.audio.output.sample_rate,
                preferred_channels=cfg.audio.output.channels,
            )
            if not args.no_audio_out
            else None
        )

        orch = GraphOrchestrator(
            qwen=qwen,
            tools_cfg=cfg.tools,
            mcp_servers=mcp_servers,
            audio_playback=playback,
            enable_speak_audio=not args.no_audio_out,
        )

        out = orch.run_turn_audio_sync(audio=seg, channels=cfg.audio.input.channels, user_text="")
        log.info(
            "voice_turn_output",
            assistant_text=out.assistant_text,
            tool_results=[r.__dict__ for r in out.tool_results],
        )
        return 0

    if args.listen:
        cap = MicrophoneAudioCapture(
            device=cfg.audio.input.device,
            sample_rate=cfg.audio.input.sample_rate,
            channels=cfg.audio.input.channels,
            vad=vad,
        )
        try:
            seg = cap.next_segment()
        finally:
            cap.close()

        log.info(
            "listen_done",
            duration_ms=seg.duration_ms,
            sample_rate=seg.sample_rate,
            bytes=len(seg.pcm),
        )

        if args.listen_save:
            _save_wav(
                pcm=seg.pcm,
                path=str(args.listen_save),
                sample_rate=seg.sample_rate,
                channels=cfg.audio.input.channels,
            )
            log.info("listen_saved", path=str(args.listen_save))

        return 0

    if args.fake:
        qwen = FakeQwenClient()
    else:
        qwen = QwenClient(cfg.qwen)

    mcp_servers = dict(cfg.mcp.servers) if cfg.mcp.enabled else None
    orch = GraphOrchestrator(qwen=qwen, tools_cfg=cfg.tools, mcp_servers=mcp_servers)
    out = orch.run_turn_text_sync(args.text)

    log.info(
        "turn_output",
        assistant_text=out.assistant_text,
        tool_results=[r.__dict__ for r in out.tool_results],
    )
    return 0
