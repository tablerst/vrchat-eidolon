from __future__ import annotations

import secrets


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_session_id() -> str:
    return secrets.token_hex(12)
