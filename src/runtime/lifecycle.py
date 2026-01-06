"""Process lifecycle and main entrypoint.

This module is the primary executable entry for `uv run vrchat-eidolon`.
It will eventually:
- handle signals and graceful shutdown
- start the outer event loop (workers + queue)
- run the inner LangGraph tick per batch of events
"""


def main() -> None:
    """Console entrypoint placeholder."""

    raise NotImplementedError("Runtime lifecycle is not implemented yet.")
