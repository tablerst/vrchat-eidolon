from __future__ import annotations

import base64
from typing import Any, Iterable

from core.config import QwenConfig
from core.types import PlanResult
from observability.logging import get_logger

from .tool_call_accumulator import ToolCallAccumulator


class QwenClient:
    """DashScope OpenAI-compat adapter (minimal interface).

    Constraints:
    - must use stream=True
    - PLAN: modalities=["text"]
    - SPEAK: modalities=["text","audio"], streaming text + audio(base64)
    """

    def __init__(self, cfg: QwenConfig) -> None:
        self._cfg = cfg
        self._log = get_logger(__name__)

    def plan(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        stream: bool = True,
        modalities: list[str] | None = None,
    ) -> PlanResult:
        if stream is not True:
            raise ValueError("Qwen-Omni constraint: stream=True is required")

        modalities = modalities or ["text"]

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'openai'. Install it with uv add openai") from e

        client = OpenAI(api_key=self._cfg.api_key, base_url=self._cfg.base_url)

        acc = ToolCallAccumulator()
        assistant_text_parts: list[str] = []
        raw_events: list[Any] = []

        stream_iter = client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
            extra_body={"modalities": modalities},
        )

        for ev in stream_iter:
            raw_events.append(ev)

            choices = getattr(ev, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                assistant_text_parts.append(content)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if not delta_tool_calls:
                continue

            normalized: list[dict[str, Any]] = []
            for tc in delta_tool_calls:
                tc_id = None
                fn_name = None
                fn_args = None

                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    fn = tc.get("function") or {}
                    if isinstance(fn, dict):
                        fn_name = fn.get("name")
                        fn_args = fn.get("arguments")
                else:
                    tc_id = getattr(tc, "id", None)
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        fn_name = getattr(fn, "name", None)
                        fn_args = getattr(fn, "arguments", None)

                if not tc_id:
                    continue

                fn_dict: dict[str, Any] = {}
                if isinstance(fn_name, str) and fn_name:
                    fn_dict["name"] = fn_name
                if isinstance(fn_args, str) and fn_args:
                    fn_dict["arguments"] = fn_args

                normalized.append({"id": str(tc_id), "function": fn_dict})

            if normalized:
                acc.add_delta(normalized)

        tool_calls = acc.get_complete_calls()
        self._log.info(
            "qwen_plan_complete",
            assistant_text_len=len("".join(assistant_text_parts)),
            tool_calls=len(tool_calls),
        )

        return PlanResult(
            tool_calls=tool_calls,
            assistant_text="".join(assistant_text_parts),
            raw_events=raw_events,
        )

    def speak(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = True,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        if stream is not True:
            raise ValueError("Qwen-Omni constraint: stream=True is required")

        modalities = modalities or ["text", "audio"]
        if "audio" in modalities:
            audio = audio or {"voice": "Chelsie", "format": "wav"}

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency 'openai'. Install it with uv add openai") from e

        client = OpenAI(api_key=self._cfg.api_key, base_url=self._cfg.base_url)

        extra_body: dict[str, Any] = {"modalities": modalities}
        if "audio" in modalities:
            extra_body["audio"] = audio

        stream_iter = client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
        )

        for ev in stream_iter:
            choices = getattr(ev, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            text = getattr(delta, "content", None)
            if isinstance(text, str) and text:
                yield {"type": "text", "delta": text}

            audio_obj = getattr(delta, "audio", None)
            if audio_obj is None:
                continue

            b64 = getattr(audio_obj, "data", None) or getattr(audio_obj, "b64", None)
            if isinstance(b64, (bytes, bytearray)):
                b64 = base64.b64encode(b64).decode("ascii")
            if isinstance(b64, str) and b64:
                yield {"type": "audio", "delta": b64}


class FakeQwenClient(QwenClient):
    """Offline stub for running the orchestrator without network/API."""

    def __init__(self) -> None:
        super().__init__(QwenConfig(api_key="k_fake"))

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
        return PlanResult(tool_calls=[], assistant_text="(fake) planning...", raw_events=None)

    def speak(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = True,
        modalities: list[str] | None = None,
        audio: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        _ = (messages, stream, modalities, audio)
        yield {"type": "text", "delta": "(fake) 你好！"}
