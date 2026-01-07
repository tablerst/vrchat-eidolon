"""Configuration loading and schema.

- YAML-first configuration under configs/*.yaml
- Strict ${ENV_VAR} expansion (missing/empty env vars are errors)
"""

from __future__ import annotations

from vrchat_eidolon.config.errors import ConfigError
from vrchat_eidolon.config.loader import load_config

__all__ = ["ConfigError", "load_config"]
