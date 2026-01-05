from __future__ import annotations


class EidolonError(Exception):
    """Base exception for this project."""


class ConfigError(EidolonError):
    """Raised when configuration is invalid or incomplete."""

    def __init__(self, message: str, *, path: str | None = None):
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
