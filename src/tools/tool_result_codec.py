from __future__ import annotations

import json
from typing import Any


def _is_json_primitive(obj: Any) -> bool:
    return obj is None or isinstance(obj, (str, int, float, bool))


def _is_json_friendly(obj: Any) -> bool:
    if _is_json_primitive(obj):
        return True
    if isinstance(obj, list):
        return all(_is_json_friendly(v) for v in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_friendly(v) for k, v in obj.items())
    return False


def make_payload(*, text: str, data: Any, raw: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Create the canonical ToolResult.content payload.

    Contract:
    - Always returns a JSON-friendly dict with keys: text/data/raw/meta.
    - `text` should be human/model-readable.
    - `data` should be the main structured output (JSON-friendly).
    - `raw` should preserve the original output as much as possible.
    - `meta` contains diagnostic and routing info.

    All code comments must be in English.
    """

    payload: dict[str, Any] = {
        "text": str(text or ""),
        "data": data if _is_json_friendly(data) else {"value": repr(data)},
        "raw": raw if _is_json_friendly(raw) else repr(raw),
        "meta": meta if _is_json_friendly(meta) else {"value": repr(meta)},
    }
    return payload


def payload_from_output(output: Any, *, meta: dict[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        return make_payload(text=output, data={"text": output}, raw=output, meta=meta)

    if _is_json_friendly(output):
        # Best-effort readable text for the model; keep short to avoid flooding.
        try:
            text = json.dumps(output, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            text = repr(output)
        if len(text) > 2000:
            text = text[:2000] + "..."
        return make_payload(text=text, data=output, raw=output, meta=meta)

    # Unknown/complex object
    return make_payload(text=repr(output), data={"value": repr(output)}, raw=output, meta=meta)


def normalize_error(*, error_type: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {
        "type": str(error_type),
        "message": str(message),
        "details": details or {},
    }

    # Backward-compatible flat detail_* fields.
    for k, v in (details or {}).items():
        err[f"detail_{k}"] = str(v)

    return err


def dumps_payload(payload: dict[str, Any]) -> str:
    """Serialize payload for OpenAI tool message content.

    Content should always be a JSON string, never a Python repr.
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
