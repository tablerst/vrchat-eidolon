from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .errors import ConfigError

# re-export for contract/tests
__all__ = [
    "AppConfig",
    "AudioConfig",
    "AudioDeviceConfig",
    "ConfigError",
    "McpConfig",
    "OscConfig",
    "QwenConfig",
    "ToolsConfig",
    "VadConfig",
    "VrchatConfig",
    "load_config",
]


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_in_str(value: str, *, path: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in os.environ or os.environ[key] == "":
            raise ConfigError(f"环境变量 {key!r} 未设置", path=path)
        return os.environ[key]

    return _ENV_PATTERN.sub(repl, value)


def _expand_env(obj: Any, *, path: str) -> Any:
    if isinstance(obj, str):
        return _expand_env_in_str(obj, path=path)
    if isinstance(obj, list):
        return [_expand_env(v, path=path) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v, path=f"{path}.{k}" if path else str(k)) for k, v in obj.items()}
    return obj


def _require(d: dict[str, Any], key: str, *, path: str) -> Any:
    if key not in d:
        raise ConfigError("缺少必填字段", path=f"{path}.{key}" if path else key)
    return d[key]


@dataclass(frozen=True)
class QwenConfig:
    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-omni-flash"
    timeout_s: float = 30.0
    max_retries: int = 3


@dataclass(frozen=True)
class OscConfig:
    send_host: str = "127.0.0.1"
    send_port: int = 9000
    recv_host: str = "0.0.0.0"
    recv_port: int = 9001
    config_path: str | None = None


@dataclass(frozen=True)
class VrchatConfig:
    osc: OscConfig = field(default_factory=OscConfig)


@dataclass(frozen=True)
class AudioDeviceConfig:
    device: str | None = None
    sample_rate: int = 48000
    channels: int = 1


@dataclass(frozen=True)
class VadConfig:
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 300
    energy_threshold: float = 0.02


@dataclass(frozen=True)
class AudioConfig:
    input: AudioDeviceConfig = field(default_factory=AudioDeviceConfig)
    output: AudioDeviceConfig = field(default_factory=AudioDeviceConfig)
    vad: VadConfig = field(default_factory=VadConfig)


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = True
    max_calls_per_turn: int = 5
    max_concurrency: int = 5
    whitelist: list[str] = field(default_factory=list)
    rate_limit: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class McpConfig:
    """Client-side MCP configuration.

    This project uses `langchain-mcp-adapters` to manage MCP connections.
    """

    enabled: bool = False
    # New preferred format: a dict of server_name -> server_config.
    # Each server config is passed to MultiServerMCPClient as-is.
    servers: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    qwen: QwenConfig
    vrchat: VrchatConfig = field(default_factory=VrchatConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and expand ${ENV_VAR}.

    Contract: plans/00-shared/02-interfaces.md
    """

    # Local dev: allow injecting secrets from .env (do not commit it).
    load_dotenv(override=False)

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError("配置文件不存在", path=str(config_path))

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"YAML 解析失败: {e}", path=str(config_path)) from e

    if not isinstance(raw, dict):
        raise ConfigError("顶层必须是 YAML mapping (dict)", path=str(config_path))

    expanded = _expand_env(raw, path="")

    qwen_raw = expanded.get("qwen", {})
    if qwen_raw is None:
        qwen_raw = {}
    if not isinstance(qwen_raw, dict):
        raise ConfigError("必须是 dict", path="qwen")

    # Contract: api_key can default from env.
    api_key = qwen_raw.get("api_key")
    if api_key is None or api_key == "":
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ConfigError("必须是非空字符串（或设置 DASHSCOPE_API_KEY）", path="qwen.api_key")

    qwen = QwenConfig(
        api_key=api_key,
        base_url=str(qwen_raw.get("base_url", QwenConfig.base_url)),
        model=str(qwen_raw.get("model", QwenConfig.model)),
        timeout_s=float(qwen_raw.get("timeout_s", QwenConfig.timeout_s)),
        max_retries=int(qwen_raw.get("max_retries", QwenConfig.max_retries)),
    )

    vrchat = VrchatConfig()
    vrchat_raw = expanded.get("vrchat")
    if isinstance(vrchat_raw, dict):
        osc_raw = vrchat_raw.get("osc")
        if isinstance(osc_raw, dict):
            vrchat = VrchatConfig(
                osc=OscConfig(
                    send_host=str(osc_raw.get("send_host", OscConfig.send_host)),
                    send_port=int(osc_raw.get("send_port", OscConfig.send_port)),
                    recv_host=str(osc_raw.get("recv_host", OscConfig.recv_host)),
                    recv_port=int(osc_raw.get("recv_port", OscConfig.recv_port)),
                    config_path=(
                        str(osc_raw["config_path"])
                        if "config_path" in osc_raw and osc_raw["config_path"] is not None
                        else None
                    ),
                )
            )

    audio = AudioConfig()
    audio_raw = expanded.get("audio")
    if isinstance(audio_raw, dict):
        in_raw = audio_raw.get("input") if isinstance(audio_raw.get("input"), dict) else {}
        out_raw = audio_raw.get("output") if isinstance(audio_raw.get("output"), dict) else {}
        vad_raw = audio_raw.get("vad") if isinstance(audio_raw.get("vad"), dict) else {}

        audio = AudioConfig(
            input=AudioDeviceConfig(
                device=(str(in_raw["device"]) if "device" in in_raw and in_raw["device"] is not None else None),
                sample_rate=int(in_raw.get("sample_rate", AudioDeviceConfig.sample_rate)),
                channels=int(in_raw.get("channels", AudioDeviceConfig.channels)),
            ),
            output=AudioDeviceConfig(
                device=(
                    str(out_raw["device"]) if "device" in out_raw and out_raw["device"] is not None else None
                ),
                sample_rate=int(out_raw.get("sample_rate", AudioDeviceConfig.sample_rate)),
                channels=int(out_raw.get("channels", AudioDeviceConfig.channels)),
            ),
            vad=VadConfig(
                silence_duration_ms=int(vad_raw.get("silence_duration_ms", VadConfig.silence_duration_ms)),
                min_speech_duration_ms=int(vad_raw.get("min_speech_duration_ms", VadConfig.min_speech_duration_ms)),
                energy_threshold=float(vad_raw.get("energy_threshold", VadConfig.energy_threshold)),
            ),
        )

    tools = ToolsConfig()
    tools_raw = expanded.get("tools")
    if isinstance(tools_raw, dict):
        whitelist = tools_raw.get("whitelist", [])
        if not isinstance(whitelist, list) or not all(isinstance(x, str) for x in whitelist):
            raise ConfigError("必须是字符串列表", path="tools.whitelist")
        rate_limit = tools_raw.get("rate_limit", {})
        if not isinstance(rate_limit, dict) or not all(isinstance(k, str) for k in rate_limit.keys()):
            raise ConfigError("必须是 dict[str,int]", path="tools.rate_limit")

        tools = ToolsConfig(
            enabled=bool(tools_raw.get("enabled", ToolsConfig.enabled)),
            max_calls_per_turn=int(tools_raw.get("max_calls_per_turn", ToolsConfig.max_calls_per_turn)),
            max_concurrency=int(tools_raw.get("max_concurrency", ToolsConfig.max_concurrency)),
            whitelist=list(whitelist),
            rate_limit={k: int(v) for k, v in rate_limit.items()},
        )

        if tools.max_concurrency < 1:
            raise ConfigError("必须是 >= 1 的整数", path="tools.max_concurrency")

    mcp = McpConfig()
    mcp_raw = expanded.get("mcp")
    if isinstance(mcp_raw, dict):
        enabled = bool(mcp_raw.get("enabled", McpConfig.enabled))
        servers_raw = mcp_raw.get("servers", {})
        if servers_raw is None:
            servers_raw = {}
        if not isinstance(servers_raw, dict):
            raise ConfigError("必须是 dict[str,dict]", path="mcp.servers")

        servers: dict[str, dict[str, Any]] = {}
        for k, v in servers_raw.items():
            if not isinstance(k, str) or not k:
                raise ConfigError("server name 必须是非空字符串", path="mcp.servers")
            if not isinstance(v, dict):
                raise ConfigError("server config 必须是 dict", path=f"mcp.servers.{k}")
            servers[k] = dict(v)

        if enabled and not servers:
            raise ConfigError("启用 MCP 时必须提供 mcp.servers", path="mcp.servers")

        if enabled:
            for name, scfg in servers.items():
                t = str(scfg.get("transport", ""))
                if t not in {"stdio", "streamable_http", "http"}:
                    raise ConfigError(f"不支持的 transport: {t!r}", path=f"mcp.servers.{name}.transport")
                if t == "stdio":
                    cmd = scfg.get("command")
                    if not isinstance(cmd, str) or not cmd.strip():
                        raise ConfigError("stdio 需要 command", path=f"mcp.servers.{name}.command")
                    args = scfg.get("args", [])
                    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
                        raise ConfigError("必须是字符串列表", path=f"mcp.servers.{name}.args")
                else:
                    u = scfg.get("url")
                    if not isinstance(u, str) or not u.strip():
                        raise ConfigError("http 需要 url", path=f"mcp.servers.{name}.url")

        mcp = McpConfig(enabled=enabled, servers=servers)

    return AppConfig(qwen=qwen, vrchat=vrchat, audio=audio, tools=tools, mcp=mcp)
