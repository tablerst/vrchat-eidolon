from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_config
from core.errors import ConfigError


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text.lstrip(), encoding="utf-8")
    return p


def test_mcp_enabled_requires_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
""",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "mcp.servers" in str(ei.value)


def test_mcp_disabled_allows_empty_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: false
""",
    )

    cfg = load_config(p)
    assert cfg.mcp.enabled is False
    assert cfg.mcp.servers == {}


def test_mcp_server_streamable_http_requires_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  servers:
    vrchat:
      transport: streamable_http
""",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "mcp.servers.vrchat.url" in str(ei.value)


def test_mcp_server_stdio_requires_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  servers:
    local:
      transport: stdio
""",
    )

    with pytest.raises(ConfigError) as ei:
        load_config(p)

    assert "mcp.servers.local.command" in str(ei.value)


def test_mcp_server_http_alias_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k_test")

    p = _write(
        tmp_path,
        """
qwen:
  api_key: ${DASHSCOPE_API_KEY}
mcp:
  enabled: true
  servers:
    vrchat:
      transport: http
      url: http://127.0.0.1:8001/mcp
""",
    )

    cfg = load_config(p)
    assert cfg.mcp.enabled is True
    assert cfg.mcp.servers["vrchat"]["transport"] == "http"
