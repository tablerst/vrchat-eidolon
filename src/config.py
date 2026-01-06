"""Configuration loader (YAML-first + strict env expansion).

This module provides a small, dependency-light config loader:
- YAML is the primary source of truth.
- Environment variables are for secrets and machine-specific overrides.

Env expansion syntax:
  - `${ENV_VAR}` inside YAML string values.
  - Expansion is strict: missing or empty env values raise ConfigError.

The repository root `.env` file is loaded automatically when present.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class ConfigError(RuntimeError):
	"""Raised when config loading or validation fails."""

	message: str

	def __str__(self) -> str:  # pragma: no cover
		return self.message


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _repo_root_from_config_path(config_path: Path) -> Path:
	# Convention: configs live under <repo_root>/configs/*.yaml
	for parent in [config_path.parent, *config_path.parents]:
		if (parent / "pyproject.toml").exists():
			return parent
	return config_path.parent


def _load_dotenv_if_present(repo_root: Path) -> None:
	env_path = repo_root / ".env"
	if env_path.exists():
		# Do not override already-set environment variables.
		load_dotenv(env_path, override=False)


def _expand_env_in_string(value: str, *, key_path: str) -> str:
	def _replace(match: re.Match[str]) -> str:
		var = match.group(1)
		resolved = os.environ.get(var)
		if resolved is None or resolved == "":
			raise ConfigError(
				f"Missing or empty environment variable '{var}' required by config at '{key_path}'."
			)
		return resolved

	return _ENV_PATTERN.sub(_replace, value)


def _expand_env(obj: Any, *, key_path: str) -> Any:
	if isinstance(obj, dict):
		return {
			k: _expand_env(v, key_path=f"{key_path}.{k}" if key_path else str(k))
			for k, v in obj.items()
		}
	if isinstance(obj, list):
		return [_expand_env(v, key_path=f"{key_path}[{i}]") for i, v in enumerate(obj)]
	if isinstance(obj, str):
		return _expand_env_in_string(obj, key_path=key_path)
	return obj


def load_config(config_path: str | Path) -> dict[str, Any]:
	"""Load a YAML config file and expand `${ENV_VAR}` placeholders.

	Args:
		config_path: Path to a YAML file.

	Returns:
		Parsed config as a dict.

	Raises:
		ConfigError: If the file is missing, invalid, or references missing env vars.
	"""

	path = Path(config_path).expanduser().resolve()
	if not path.exists():
		raise ConfigError(f"Config file not found: {path}")

	repo_root = _repo_root_from_config_path(path)
	_load_dotenv_if_present(repo_root)

	try:
		raw = yaml.safe_load(path.read_text(encoding="utf-8"))
	except Exception as exc:  # pragma: no cover
		raise ConfigError(f"Failed to parse YAML config at {path}: {exc}") from exc

	if raw is None:
		raise ConfigError(f"Config file is empty: {path}")
	if not isinstance(raw, dict):
		raise ConfigError(f"Config root must be a mapping/object: {path}")

	expanded = _expand_env(raw, key_path="")
	if not isinstance(expanded, dict):
		raise ConfigError(f"Expanded config root must be a mapping/object: {path}")

	return expanded
