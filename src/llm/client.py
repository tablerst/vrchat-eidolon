"""OpenAI-compatible LLM client wrapper.

This client uses LangChain's OpenAI wrapper (`langchain_openai.ChatOpenAI`).

Hard constraints for Qwen Omni models:
- Streaming is mandatory (`stream=True`).
- Input modality constraint: in a single user message, content can be text + one other modality.
- This project defaults to thinking disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llm.tool_call_accumulator import InvalidToolCall, ToolCall, ToolCallAccumulator


@dataclass(frozen=True)
class PlanResult:
	assistant_text: str
	tool_calls: list[ToolCall]
	invalid_tool_calls: list[InvalidToolCall]


@dataclass(frozen=True)
class SpeakEvent:
	"""A streamed SPEAK event.

	type:
		- "text": delta contains a text token/fragment
		- "audio": delta contains a base64-encoded audio chunk (format is best-effort)
	"""

	type: str
	delta: str
	format: str | None = None


class EidolonLLMClient:
	"""LLM client for PLAN and SPEAK phases."""

	def __init__(
		self,
		*,
		api_key: str,
		base_url: str,
		model: str,
		timeout_s: float = 30,
		max_retries: int = 3,
	) -> None:
		# NOTE: `streaming=True` is required for Qwen Omni.
		secret_key = SecretStr(api_key)

		self._plan_model = ChatOpenAI(
			model=model,
			api_key=secret_key,
			base_url=base_url,
			timeout=timeout_s,
			max_retries=max_retries,
			streaming=True,
			model_kwargs={"modalities": ["text"]},
		)
		self._speak_model = ChatOpenAI(
			model=model,
			api_key=secret_key,
			base_url=base_url,
			timeout=timeout_s,
			max_retries=max_retries,
			streaming=True,
			model_kwargs={"modalities": ["text", "audio"]},
		)

	@classmethod
	def from_config(cls, cfg: dict[str, Any]) -> "EidolonLLMClient":
		qwen = cfg.get("qwen")
		if not isinstance(qwen, dict):
			raise ValueError("Missing 'qwen' config mapping")

		api_key = qwen.get("api_key")
		base_url = qwen.get("base_url")
		model = qwen.get("model")
		timeout_s = qwen.get("timeout_s", 30)
		max_retries = qwen.get("max_retries", 3)

		if not isinstance(api_key, str) or not api_key:
			raise ValueError("qwen.api_key must be a non-empty string")
		if not isinstance(base_url, str) or not base_url:
			raise ValueError("qwen.base_url must be a non-empty string")
		if not isinstance(model, str) or not model:
			raise ValueError("qwen.model must be a non-empty string")

		return cls(
			api_key=api_key,
			base_url=base_url,
			model=model,
			timeout_s=float(timeout_s),
			max_retries=int(max_retries),
		)

	def plan(
		self,
		messages: list[BaseMessage],
		*,
		tools: list[Any] | None = None,
	) -> PlanResult:
		"""Run PLAN (text-only output; may include tool calls).

		Notes:
		- Tool-call arguments are accumulated from streaming chunks.
		- Invalid tool calls are returned in invalid_tool_calls (non-fatal).
		"""

		model = self._plan_model.bind_tools(tools) if tools else self._plan_model
		text_parts: list[str] = []
		accumulator = ToolCallAccumulator()

		for chunk in model.stream(messages):
			# LangChain streams AIMessageChunk objects.
			content = getattr(chunk, "content", None)
			if isinstance(content, str) and content:
				text_parts.append(content)

			tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
			if isinstance(tool_call_chunks, list) and tool_call_chunks:
				accumulator.add_chunks(tool_call_chunks)

		tool_calls, invalid = accumulator.finalize()
		return PlanResult(
			assistant_text="".join(text_parts),
			tool_calls=tool_calls,
			invalid_tool_calls=invalid,
		)

	def speak_stream(self, messages: list[BaseMessage]) -> Iterator[SpeakEvent]:
		"""Stream SPEAK output (text + best-effort audio deltas)."""

		for chunk in self._speak_model.stream(messages):
			content = getattr(chunk, "content", None)
			if isinstance(content, str) and content:
				yield SpeakEvent(type="text", delta=content)

			audio_delta, audio_format = _extract_audio_delta(chunk)
			if audio_delta:
				yield SpeakEvent(type="audio", delta=audio_delta, format=audio_format)


def _extract_audio_delta(chunk: Any) -> tuple[str | None, str | None]:
	"""Best-effort extraction of audio deltas from a streamed LangChain chunk.

	Different OpenAI-compatible providers may attach audio data under different keys.
	We intentionally keep this tolerant and non-fatal.
	"""

	additional = getattr(chunk, "additional_kwargs", None)
	meta = getattr(chunk, "response_metadata", None)

	for container in (additional, meta):
		if not isinstance(container, dict):
			continue
		audio_obj = container.get("audio")
		if isinstance(audio_obj, str) and audio_obj:
			return audio_obj, None
		if isinstance(audio_obj, dict):
			# Common shapes: {"data": "<b64>", "format": "wav"}
			data = audio_obj.get("data") or audio_obj.get("delta") or audio_obj.get("b64")
			fmt = audio_obj.get("format")
			if isinstance(data, str) and data:
				return data, fmt if isinstance(fmt, str) else None

	return None, None
