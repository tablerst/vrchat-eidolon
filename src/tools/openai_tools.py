from __future__ import annotations

from typing import Any


def get_openai_tool_specs(names: list[str]) -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool specs for a known set of tools.

    MVP-0 intentionally exposes a small, hand-authored set of tools.

    Note: OpenAI-compatible format:
    {
      "type": "function",
      "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": { ...JSON Schema... }
      }
    }
    """

    out: list[dict[str, Any]] = []
    for name in names:
        out.append(_spec_for(name))
    return out


def _spec_for(name: str) -> dict[str, Any]:
    if name == "vrc_status":
        return {
            "type": "function",
            "function": {
                "name": "vrc_status",
                "description": "Get VRChat/MCP server status (read-only).",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }

    if name == "vrc_chat_send":
        return {
            "type": "function",
            "function": {
                "name": "vrc_chat_send",
                "description": "Send a message to the VRChat chatbox.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Message to send."},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }

    raise KeyError(f"unknown tool spec: {name}")
