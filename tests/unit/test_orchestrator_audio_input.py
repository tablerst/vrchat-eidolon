from __future__ import annotations

from typing import Any

from audio.types import AudioSegment
from core.config import ToolsConfig
from core.types import PlanResult
from orchestrator.graph_orchestrator import GraphOrchestrator
from qwen.client import FakeQwenClient


class RecordingQwen(FakeQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.last_plan_messages: list[dict[str, Any]] | None = None
        self.last_plan_modalities: list[str] | None = None
        self.last_plan_tools: list[dict[str, Any]] | None = None
        self.last_plan_tool_choice: str | dict[str, Any] | None = None

    def plan(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = True,
        modalities: list[str] | None = None,
    ) -> PlanResult:
        _ = stream
        self.last_plan_messages = messages
        self.last_plan_modalities = modalities
        self.last_plan_tools = tools
        self.last_plan_tool_choice = tool_choice
        return PlanResult(tool_calls=[], assistant_text="(test) planning", raw_events=None)

    def speak(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = True,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
    ):
        _ = (messages, stream, modalities, audio)
        yield {"type": "text", "delta": "(test) reply"}


def test_run_turn_audio_builds_input_audio_message() -> None:
    qwen = RecordingQwen()
    orch = GraphOrchestrator(
        qwen=qwen,
        tools_cfg=ToolsConfig(enabled=True, whitelist=[], rate_limit={}),
        mcp_servers=None,
    )

    seg = AudioSegment(pcm=b"\x00\x00" * 4800, sample_rate=48_000, duration_ms=50)
    out = orch.run_turn_audio_sync(audio=seg, channels=1, user_text="")

    assert "(test) reply" in out.assistant_text

    assert qwen.last_plan_tools is None
    assert qwen.last_plan_tool_choice == "none"
    assert qwen.last_plan_modalities == ["text"]

    assert qwen.last_plan_messages is not None
    assert len(qwen.last_plan_messages) == 1

    msg = qwen.last_plan_messages[0]
    assert msg.get("role") == "user"

    content = msg.get("content")
    assert isinstance(content, list)
    assert content

    audio_part = content[0]
    assert audio_part.get("type") == "input_audio"
    input_audio = audio_part.get("input_audio")
    assert isinstance(input_audio, dict)
    assert input_audio.get("format") == "wav"

    data = input_audio.get("data")
    assert isinstance(data, str)
    assert data.startswith("data:;base64,")
