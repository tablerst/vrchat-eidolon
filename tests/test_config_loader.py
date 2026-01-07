from __future__ import annotations

from pathlib import Path

import pytest

from vrchat_eidolon.config.errors import ConfigError
from vrchat_eidolon.config.loader import load_config


def test_load_config_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "abc123")

    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
  base_url: https://example.invalid
nested:
  arr:
    - hi-${DASHSCOPE_API_KEY}
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path, load_dotenv_file=False)
    assert cfg["qwen"]["api_key"] == "abc123"
    assert cfg["nested"]["arr"][0] == "hi-abc123"


def test_load_config_missing_env_var_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, load_dotenv_file=False)

    msg = str(ei.value)
    assert "DASHSCOPE_API_KEY" in msg
    assert "missing" in msg


def test_load_config_empty_env_var_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")

    cfg_path = tmp_path / "app.yaml"
    cfg_path.write_text(
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(cfg_path, load_dotenv_file=False)

    msg = str(ei.value)
    assert "DASHSCOPE_API_KEY" in msg
    assert "empty" in msg
