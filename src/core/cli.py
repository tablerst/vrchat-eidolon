from __future__ import annotations

import argparse
import os
import wave
from pathlib import Path

from audio import MicrophoneAudioCapture
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

    if args.listen:
        cap = MicrophoneAudioCapture(
            device=cfg.audio.input.device,
            sample_rate=cfg.audio.input.sample_rate,
            channels=cfg.audio.input.channels,
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
