from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AppConfig:
    """A thin config wrapper.

    Milestone 0 keeps the schema flexible; later milestones can evolve this into
    a stricter typed model.
    """

    raw: Mapping[str, Any]

    @property
    def qwen_api_key(self) -> str | None:
        qwen = self.raw.get("qwen")
        if not isinstance(qwen, Mapping):
            return None
        api_key = qwen.get("api_key")
        return api_key if isinstance(api_key, str) else None
