from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class McpNameMaps:
    server_to_prefix: dict[str, str]
    prefix_to_server: dict[str, str]


def _prefix_from_key(key: str) -> str:
    """Generate a 4-char prefix from a server_key.

    Requirements:
    - length == 4
    - first character must not be a digit
    - stable across processes
    """

    # 16-bit digest is enough to derive a short stable prefix.
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=2).digest()
    x = int.from_bytes(digest, "big")

    first = chr(ord("a") + (x % 26))
    tail = f"{x:04x}"[1:]  # last 3 hex chars
    return first + tail


def build_name_maps(server_keys: list[str]) -> McpNameMaps:
    used: set[str] = set()
    server_to_prefix: dict[str, str] = {}

    for sk in server_keys:
        i = 0
        while True:
            candidate = _prefix_from_key(sk if i == 0 else f"{sk}#{i}")
            if candidate not in used:
                used.add(candidate)
                server_to_prefix[sk] = candidate
                break
            i += 1

    prefix_to_server = {p: s for s, p in server_to_prefix.items()}
    return McpNameMaps(server_to_prefix=server_to_prefix, prefix_to_server=prefix_to_server)


def to_model_tool_name(*, server_key: str, tool_name: str, maps: McpNameMaps, multi: bool) -> str:
    if not multi:
        return tool_name
    prefix = maps.server_to_prefix[server_key]
    return f"{prefix}__{tool_name}"


def parse_model_tool_name(*, model_tool_name: str, maps: McpNameMaps, multi: bool, single_server_key: str) -> tuple[str, str]:
    if not multi:
        return single_server_key, model_tool_name

    if "__" not in model_tool_name:
        raise ValueError("missing '__' in prefixed tool name")

    prefix, tool_name = model_tool_name.split("__", 1)
    if not prefix or not tool_name:
        raise ValueError("invalid prefixed tool name")

    server_key = maps.prefix_to_server.get(prefix)
    if server_key is None:
        raise ValueError(f"unknown tool prefix: {prefix!r}")

    return server_key, tool_name
