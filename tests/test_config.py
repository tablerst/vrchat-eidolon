from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.config import load_config
from core.errors import ConfigError


def test_load_config_expands_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")
    monkeypatch.setenv("APPDATA", "C:/Users/me/AppData/Roaming")

    p = tmp_path / "app.yaml"
    p.write_text(
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
vrchat:
  osc:
    config_path: ${APPDATA}/../LocalLow/VRChat/VRChat/OSC
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(p)
    assert cfg.qwen.api_key == "k_test"
    assert "LocalLow/VRChat/VRChat/OSC" in (cfg.vrchat.osc.config_path or "")


def test_load_config_missing_env_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    p = tmp_path / "app.yaml"
    p.write_text(
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "DASHSCOPE_API_KEY" in str(ei.value)


def test_repo_configs_app_yaml_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    # 仅验证语法 + 展开机制；不要求真实 Key。
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_dummy")
    monkeypatch.setenv("APPDATA", os.environ.get("APPDATA", "C:/Users/me/AppData/Roaming"))

    cfg = load_config(Path("configs/app.yaml"))
    assert cfg.qwen.model
