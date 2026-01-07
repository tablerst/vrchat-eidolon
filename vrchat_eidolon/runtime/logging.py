"""Structured logging setup.

This module configures standard-library logging with stable context fields.

Required fields injected into every log record:
- trace_id
- session_id
- turn_id
- state
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing import Any


_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="0")
_state_var: contextvars.ContextVar[str] = contextvars.ContextVar("state", default="BOOT")


def init_observability_context(*, trace_id: str | None = None, session_id: str | None = None) -> None:
	"""Initialize trace/session IDs for the current process."""

	_trace_id_var.set(trace_id or uuid.uuid4().hex)
	_session_id_var.set(session_id or uuid.uuid4().hex)


def set_turn_id(turn_id: int) -> None:
	_turn_id_var.set(str(int(turn_id)))


def set_state(state: str) -> None:
	_state_var.set(state)


def get_context_fields() -> dict[str, str]:
	return {
		"trace_id": _trace_id_var.get(),
		"session_id": _session_id_var.get(),
		"turn_id": _turn_id_var.get(),
		"state": _state_var.get(),
	}


class _ContextFilter(logging.Filter):
	def filter(self, record: logging.LogRecord) -> bool:
		ctx = get_context_fields()
		record.trace_id = ctx["trace_id"]
		record.session_id = ctx["session_id"]
		record.turn_id = ctx["turn_id"]
		record.state = ctx["state"]
		return True


def configure_logging(*, level: str = "INFO") -> None:
	"""Configure root logging once.

	Safe to call multiple times.
	"""

	root = logging.getLogger()
	if getattr(root, "_eidolon_configured", False):
		root.setLevel(level)
		return

	root.setLevel(level)

	handler = logging.StreamHandler(stream=sys.stdout)
	handler.setLevel(level)
	handler.addFilter(_ContextFilter())
	handler.setFormatter(
		logging.Formatter(
			fmt=(
				"%(asctime)s %(levelname)s "
				"trace_id=%(trace_id)s session_id=%(session_id)s "
				"turn_id=%(turn_id)s state=%(state)s "
				"%(name)s: %(message)s"
			)
		)
	)

	root.addHandler(handler)
	setattr(root, "_eidolon_configured", True)


def get_logger(name: str | None = None, **extra: Any) -> logging.LoggerAdapter:
	"""Return a logger adapter that can carry additional static fields."""

	base = logging.getLogger(name or "eidolon")
	return logging.LoggerAdapter(base, extra)
