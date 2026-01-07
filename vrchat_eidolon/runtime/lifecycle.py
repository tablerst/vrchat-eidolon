from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from vrchat_eidolon.config.errors import ConfigError
from vrchat_eidolon.config.loader import load_config, resolve_profile_configs
from vrchat_eidolon.observability.logging import configure_logging
from vrchat_eidolon.runtime.speech_loop import run_speech_loop


logger = logging.getLogger(__name__)


def _redact_secrets(obj):  # noqa: ANN001
    """Best-effort redaction for human-facing config dumps.

    We keep config loading strict and fail-fast, but `print-config` should not
    echo secrets (API keys, tokens) to stdout by default.
    """

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and any(p in k.lower() for p in ("api_key", "token", "secret", "password")):
                out[k] = "<redacted>"
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrchat-eidolon",
        description="VRChat Eidolon (Milestone 1: Realtime Speech Loop MVP)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=Path,
        help="Path to a YAML config file (skips profile resolution)",
    )
    group.add_argument(
        "--profile",
        choices=["app", "dev"],
        default="app",
        help="Config profile under ./configs (app loads app.yaml; dev overlays dev.yaml)",
    )

    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the Realtime Speech Loop")
    run_p.set_defaults(command="run")

    dev_p = sub.add_parser("devices", help="List available audio devices")
    dev_p.set_defaults(command="devices")

    print_p = sub.add_parser("print-config", help="Load and print the expanded config")
    print_p.set_defaults(command="print-config")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint referenced by pyproject.toml.

    Acceptance criteria (Milestone 0):
    - `vrchat-eidolon --help` works without requiring env vars.
    - Running without DASHSCOPE_API_KEY fails fast with a clear error.
    """

    argv_list = list(argv) if argv is not None else sys.argv[1:]

    # Default to `run` when no subcommand is provided.
    if not argv_list or (argv_list and not argv_list[0].startswith("-")):
        # If the first token is a known subcommand, leave it as-is.
        known = {"run", "print-config", "devices"}
        if not argv_list or argv_list[0] not in known:
            argv_list = ["run", *argv_list]

    parser = _build_parser()

    try:
        ns = parser.parse_args(argv_list)
    except SystemExit as e:
        # argparse has already printed help/usage to stdout/stderr.
        code = e.code
        return int(code) if isinstance(code, int) else 1

    configure_logging(level=ns.log_level)

    # Commands that should work without loading any config (and therefore
    # without requiring env vars like DASHSCOPE_API_KEY).
    if ns.command == "devices":
        import sounddevice as sd

        sys.stdout.write(json.dumps(sd.query_devices(), ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return 0

    configs_dir = Path.cwd() / "configs"
    try:
        if ns.config is not None:
            config_paths = [ns.config]
        else:
            config_paths = resolve_profile_configs(profile=ns.profile, configs_dir=configs_dir)

        cfg = load_config(config_paths)

        logger.info(
            "config_loaded",
            extra={"config_files": [str(p) for p in config_paths]},
        )

        if ns.command == "print-config":
            sys.stdout.write(json.dumps(_redact_secrets(cfg), ensure_ascii=False, indent=2))
            sys.stdout.write("\n")
            return 0

        logger.info("runtime_started", extra={"milestone": 1})
        asyncio.run(run_speech_loop(cfg))
        return 0

    except ConfigError as e:
        # Keep the message extremely clear for fast-fail debugging.
        logger.error("config_error", extra={"error": str(e)})
        sys.stderr.write(f"ConfigError: {e}\n")
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("fatal_error")
        sys.stderr.write(f"Fatal error: {e}\n")
        return 1
