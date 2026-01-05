from __future__ import annotations

from contextvars import ContextVar


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_turn_id: ContextVar[str | int | None] = ContextVar("turn_id", default=None)
_state: ContextVar[str | None] = ContextVar("state", default=None)
_errors: ContextVar[list[str] | None] = ContextVar("errors", default=None)


def bind_context(*, trace_id: str, session_id: str, turn_id: str | int) -> None:
    _trace_id.set(trace_id)
    _session_id.set(session_id)
    _turn_id.set(turn_id)
    _errors.set([])


def set_state(state: str) -> None:
    _state.set(state)


def add_error(message: str) -> None:
    errs = list(_errors.get() or [])
    errs.append(message)
    _errors.set(errs)


def snapshot() -> dict[str, object]:
    """Return a snapshot of current observability context for logging."""

    out: dict[str, object] = {}
    if (v := _trace_id.get()) is not None:
        out["trace_id"] = v
    if (v := _session_id.get()) is not None:
        out["session_id"] = v
    if (v := _turn_id.get()) is not None:
        out["turn_id"] = v
    if (v := _state.get()) is not None:
        out["state"] = v
    out["errors"] = list(_errors.get() or [])
    return out
