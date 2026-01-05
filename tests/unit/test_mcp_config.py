from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config
from core.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text.lstrip(), encoding="utf-8")
    return p


def test_mcp_http_uses_default_url_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  transport: http
""",
    )

    cfg = load_config(p)
    assert cfg.mcp.enabled is True
    assert cfg.mcp.transport == "http"
    assert cfg.mcp.url == "http://127.0.0.1:8001/mcp"
    assert cfg.mcp.servers["default"]["transport"] in {"http", "streamable_http"}
    assert cfg.mcp.servers["default"]["url"] == "http://127.0.0.1:8001/mcp"


def test_mcp_http_rejects_empty_url_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  transport: http
  url: ""
""",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "mcp.url" in str(ei.value)


def test_mcp_http_loads_with_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  transport: http
  url: http://127.0.0.1:8001/mcp
""",
    )

    cfg = load_config(p)
    assert cfg.mcp.enabled is True
    assert cfg.mcp.transport == "http"
    assert cfg.mcp.url == "http://127.0.0.1:8001/mcp"
    assert cfg.mcp.servers["default"]["url"] == "http://127.0.0.1:8001/mcp"


def test_mcp_stdio_requires_command_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  transport: stdio
""",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "mcp.command" in str(ei.value)


def test_mcp_default_transport_is_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
""",
    )

    cfg = load_config(p)
    assert cfg.mcp.transport == "http"
