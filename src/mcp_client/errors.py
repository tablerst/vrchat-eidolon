from __future__ import annotations


class McpClientError(RuntimeError):
    """Base exception for MCP client failures.

    The goal is to normalize transport and protocol failures into a small set of
    stable error types that can be mapped into ToolResult.error.
    """

    def __init__(self, error_type: str, message: str, *, details: dict[str, str] | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


class McpDependencyMissingError(McpClientError):
    def __init__(self, *, missing: str, install_hint: str):
        super().__init__(
            "dependency_missing",
            f"Missing dependency {missing!r}. {install_hint}",
            details={"missing": missing, "install_hint": install_hint},
        )


class McpTimeoutError(McpClientError):
    def __init__(self, *, timeout_s: float):
        super().__init__(
            "timeout",
            f"MCP call timed out after {timeout_s}s",
            details={"timeout_s": str(timeout_s)},
        )
