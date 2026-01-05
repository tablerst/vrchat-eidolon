from __future__ import annotations

import json
import logging
from typing import Any

from .context import snapshot


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Minimum requirements on the critical path:
        # trace_id/session_id/turn_id/state/latency_ms/errors[]
        payload.update(snapshot())

        # Convention: extra fields are carried in record.__dict__.
        for k, v in record.__dict__.items():
            if k in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = repr(v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


_configured = False


class KVLogger:
    """A tiny structured logging adapter."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self._log("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        self._log("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self._log("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self._log("error", msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        extra = kwargs.pop("extra", None)
        if extra is None:
            extra_dict: dict[str, object] = {}
        elif isinstance(extra, dict):
            extra_dict = dict(extra)
        else:
            extra_dict = {"extra": repr(extra)}

        for k, v in kwargs.items():
            extra_dict[k] = v

        self._logger.exception(msg, *args, extra=extra_dict)

    def _log(self, level: str, msg: str, *args: object, **kwargs: object) -> None:
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", None)
        if extra is None:
            extra_dict: dict[str, object] = {}
        elif isinstance(extra, dict):
            extra_dict = dict(extra)
        else:
            extra_dict = {"extra": repr(extra)}

        for k, v in kwargs.items():
            extra_dict[k] = v

        log_fn = getattr(self._logger, level)
        log_fn(msg, *args, extra=extra_dict, exc_info=exc_info, stack_info=stack_info)


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    _configured = True


def get_logger(name: str = "eidolon", *, level: str = "INFO") -> KVLogger:
    configure_logging(level=level)
    return KVLogger(logging.getLogger(name))
