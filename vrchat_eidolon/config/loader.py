from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml
from dotenv import load_dotenv

from vrchat_eidolon.config.errors import ConfigError


_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True, slots=True)
class _UnresolvedEnvRef:
    """Tracks an unresolved ${ENV_VAR} reference for better error messages."""

    var_name: str
    source_file: str
    key_path: str
    reason: str  # "missing" | "empty"


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-merge two mappings.

    - Dicts are merged recursively.
    - Other values are replaced.

    This is intentionally minimal (Milestone 0).
    """

    for k, v in overlay.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_merge(dict(base[k]), v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return yaml.safe_load(text)


def _expand_env_in_obj(
    obj: Any,
    *,
    source_file: str,
    key_path: str,
    unresolved: list[_UnresolvedEnvRef],
) -> Any:
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.getenv(name)
            if value is None:
                unresolved.append(
                    _UnresolvedEnvRef(
                        var_name=name,
                        source_file=source_file,
                        key_path=key_path,
                        reason="missing",
                    )
                )
                return match.group(0)
            if value == "":
                unresolved.append(
                    _UnresolvedEnvRef(
                        var_name=name,
                        source_file=source_file,
                        key_path=key_path,
                        reason="empty",
                    )
                )
                return match.group(0)
            return value

        return _ENV_PLACEHOLDER_RE.sub(repl, obj)

    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            child_path = f"{key_path}.{k}" if key_path else str(k)
            out[str(k)] = _expand_env_in_obj(
                v,
                source_file=source_file,
                key_path=child_path,
                unresolved=unresolved,
            )
        return out

    if isinstance(obj, list):
        out_list: list[Any] = []
        for i, v in enumerate(obj):
            child_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
            out_list.append(
                _expand_env_in_obj(
                    v,
                    source_file=source_file,
                    key_path=child_path,
                    unresolved=unresolved,
                )
            )
        return out_list

    return obj


def load_config(
    paths: Path | Sequence[Path],
    *,
    load_dotenv_file: bool = True,
    dotenv_path: Path | None = None,
) -> dict[str, Any]:
    """Load YAML config files with strict ${ENV_VAR} expansion.

    Args:
        paths: One or more YAML files. When multiple are provided, they are merged
            (later files override earlier ones).
        load_dotenv_file: Whether to load a .env file before expansion.
        dotenv_path: Optional explicit .env path. When omitted, attempts to load
            a `.env` in the current working directory.

    Raises:
        ConfigError: If YAML is invalid, or env expansion is unresolved.
    """

    file_list: list[Path] = [paths] if isinstance(paths, Path) else list(paths)
    if not file_list:
        raise ConfigError("No config files provided")

    if load_dotenv_file:
        # load_dotenv() is intentionally best-effort here; strictness is enforced
        # by the ${ENV_VAR} expansion step.
        load_dotenv(dotenv_path or Path.cwd() / ".env", override=False)

    merged: dict[str, Any] = {}
    for p in file_list:
        try:
            fragment = _load_yaml(p)
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"Failed to read YAML config: {p}: {e}") from e

        if fragment is None:
            fragment = {}

        if not isinstance(fragment, Mapping):
            raise ConfigError(f"Top-level YAML must be a mapping/dict: {p}")

        merged = _deep_merge(merged, fragment)  # type: ignore[arg-type]

    unresolved: list[_UnresolvedEnvRef] = []
    expanded = _expand_env_in_obj(
        merged,
        source_file=",".join(str(p) for p in file_list),
        key_path="",
        unresolved=unresolved,
    )

    if unresolved:
        lines: list[str] = ["Unresolved environment variables in config:"]
        for ref in unresolved:
            where = ref.key_path or "<root>"
            lines.append(
                f"- {ref.var_name} ({ref.reason}) at {where} in {ref.source_file}"
            )
        raise ConfigError("\n".join(lines))

    if not isinstance(expanded, dict):
        raise ConfigError("Expanded config must be a dict")

    return expanded


def resolve_profile_configs(*, profile: str, configs_dir: Path) -> list[Path]:
    """Resolve config file list for a given profile.

    - profile=app -> [configs/app.yaml]
    - profile=dev -> [configs/app.yaml, configs/dev.yaml]

    This keeps Milestone 0 simple while allowing overrides.
    """

    if profile == "app":
        return [configs_dir / "app.yaml"]
    if profile == "dev":
        return [configs_dir / "app.yaml", configs_dir / "dev.yaml"]
    raise ConfigError(f"Unknown profile: {profile}")
