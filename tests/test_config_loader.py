from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import ConfigError, load_config


def test_load_config_dev_defaults_domestic_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure required env vars exist for strict expansion.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("APPDATA", "C:/Users/test/AppData/Roaming")

    cfg = load_config(Path("configs") / "dev.yaml")

    assert cfg["qwen"]["base_url"].startswith("https://dashscope.aliyuncs.com/")
    assert cfg["qwen"]["model"] == "qwen3-omni-flash"

    tools = cfg["tools"]
    assert tools["enabled"] is True
    assert tools["whitelist"] == []


def test_load_config_missing_required_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force empty to avoid reliance on the developer machine environment.
    # Strict expansion treats empty as missing.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("APPDATA", "C:/Users/test/AppData/Roaming")

    with pytest.raises(ConfigError) as exc:
        load_config(Path("configs") / "app.yaml")

    assert "DASHSCOPE_API_KEY" in str(exc.value)
