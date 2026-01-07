"""Process lifecycle and main entrypoint.

This module is the primary executable entry for `uv run vrchat-eidolon`.
It will eventually:
- handle signals and graceful shutdown
- start the outer event loop (workers + queue)
- run the inner LangGraph tick per batch of events
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vrchat_eidolon.runtime.logging import (
    configure_logging,
    get_logger,
    init_observability_context,
    set_state,
    set_turn_id,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrchat-eidolon")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "app.yaml"),
        help="Path to a YAML config file (default: configs/app.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single boot cycle and exit (useful for smoke tests)",
    )
    return parser


def main() -> None:
    """Console entrypoint."""

    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    init_observability_context()
    set_turn_id(0)
    set_state("BOOT")
    log = get_logger("runtime.lifecycle")

    try:
        from vrchat_eidolon.config import ConfigError, load_config

        cfg = load_config(args.config)
    except Exception as exc:
        # Config errors are fatal at startup.
        log.error("Failed to load config", extra={"error": str(exc), "config": args.config})
        raise

    set_state("IDLE")
    log.info("Config loaded", extra={"config": args.config})

    # Build core adapters (no network calls at startup).
    try:
        from vrchat_eidolon.llm.client import EidolonLLMClient

        _llm = EidolonLLMClient.from_config(cfg)
        log.info("LLM client initialized", extra={"model": cfg.get("qwen", {}).get("model")})
    except Exception as exc:
        log.error("Failed to initialize LLM client", extra={"error": str(exc)})
        raise

    # Placeholder: the real implementation will start workers and run the graph tick.
    if args.once:
        log.info("Lifecycle placeholder completed (once)")
        return

    log.info("Lifecycle placeholder running (Ctrl+C to exit)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        set_state("SHUTDOWN")
        log.info("Shutdown requested")
