"""LLM adapters (OpenAI-compatible client, prompts)."""

from vrchat_eidolon.llm.client import EidolonLLMClient, PlanResult, SpeakEvent
from vrchat_eidolon.llm.tool_call_accumulator import InvalidToolCall, ToolCall, ToolCallAccumulator

__all__ = [
	"EidolonLLMClient",
	"PlanResult",
	"SpeakEvent",
	"ToolCall",
	"InvalidToolCall",
	"ToolCallAccumulator",
]
