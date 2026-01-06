from __future__ import annotations

from typing import Any


def tool_to_openai_spec(tool: Any, *, name_override: str) -> dict[str, Any]:
    """Convert a LangChain tool into an OpenAI-compatible tool spec.

    We prefer LangChain's own converter when available.
    """

    spec: dict[str, Any]

    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool  # type: ignore

        spec = convert_to_openai_tool(tool)  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001
        spec = _fallback_spec(tool)

    # Ensure shape and override name.
    fn = spec.get("function")
    if not isinstance(fn, dict):
        fn = {}
        spec["function"] = fn

    fn["name"] = name_override

    if "description" not in fn:
        desc = getattr(tool, "description", None)
        fn["description"] = str(desc) if desc is not None else ""

    if "parameters" not in fn or not isinstance(fn.get("parameters"), dict):
        fn["parameters"] = {"type": "object", "properties": {}, "additionalProperties": False}

    if "type" not in spec:
        spec["type"] = "function"

    return spec


def _fallback_spec(tool: Any) -> dict[str, Any]:
    desc = getattr(tool, "description", None)

    parameters: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

    schema = None
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = args_schema.model_json_schema()
        except Exception:  # noqa: BLE001
            schema = None

    if schema is None:
        get_input_schema = getattr(tool, "get_input_schema", None)
        if callable(get_input_schema):
            try:
                schema = get_input_schema().model_json_schema()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                schema = None

    if isinstance(schema, dict):
        # Best effort: ensure it's an object schema.
        if schema.get("type") == "object":
            parameters = schema

    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool, "name", "tool")),
            "description": str(desc) if desc is not None else "",
            "parameters": parameters,
        },
    }
