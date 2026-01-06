from __future__ import annotations

from typing import Any

from core.config import ToolsConfig
from core.types import PlanResult
from orchestrator.graph_orchestrator import GraphOrchestrator
from qwen.client import FakeQwenClient


class RecordingPlayback:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.chunks: list[str] = []

    def start(self) -> None:
        self.started = True

    def push_audio_b64(self, chunk_b64: str) -> None:
        self.chunks.append(chunk_b64)

    def close(self) -> None:
        self.closed = True


class RecordingQwen(FakeQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.last_speak_modalities: list[str] | None = None

    def plan(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = True,
        modalities: list[str] | None = None,
    ) -> PlanResult:
        _ = (messages, tools, tool_choice, stream, modalities)
        return PlanResult(tool_calls=[], assistant_text="(test) planning", raw_events=None)

    def speak(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = True,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
    ):
        _ = (messages, stream, audio)
        self.last_speak_modalities = modalities
        yield {"type": "text", "delta": "hello"}
        yield {"type": "audio", "delta": "QUJD"}  # base64 for 'ABC'
        yield {"type": "text", "delta": " world"}


def test_speak_audio_enables_playback() -> None:
    qwen = RecordingQwen()
    pb = RecordingPlayback()

    orch = GraphOrchestrator(
        qwen=qwen,
        tools_cfg=ToolsConfig(enabled=True, whitelist=[], rate_limit={}),
        mcp_servers=None,
        audio_playback=pb,  # type: ignore[arg-type]
        enable_speak_audio=True,
    )

    out = orch.run_turn_text_sync("hi")

    assert out.assistant_text.endswith("hello world")
    assert qwen.last_speak_modalities == ["text", "audio"]

    assert pb.started is True
    assert pb.closed is True
    assert pb.chunks == ["QUJD"]
