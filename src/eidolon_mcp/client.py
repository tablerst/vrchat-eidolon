"""MCP client lifecycle management.

This module builds a `MultiServerMCPClient` from YAML config.

The external MCP server(s) expose VRChat-related tools (OSC/chatbox/etc).
This repo acts as a client and must treat remote schemas and outputs as
untrusted input.
"""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


def build_mcp_client(cfg: dict[str, Any]) -> MultiServerMCPClient | None:
    """Build an MCP client from config.

    Returns None when MCP is disabled or no servers are configured.
    """

    mcp_cfg = cfg.get("mcp")
    if not isinstance(mcp_cfg, dict):
        return None
    if not bool(mcp_cfg.get("enabled", False)):
        return None

    servers = mcp_cfg.get("servers")
    if not isinstance(servers, dict) or not servers:
        return None

    connections: dict[str, Any] = {}
    for server_name, server_cfg in servers.items():
        if not isinstance(server_name, str) or not server_name:
            continue
        if not isinstance(server_cfg, dict):
            continue

        transport = server_cfg.get("transport")
        if transport not in {"streamable_http", "sse", "websocket", "stdio"}:
            raise ValueError(f"Unsupported MCP transport for '{server_name}': {transport!r}")

        if transport == "stdio":
            command = server_cfg.get("command")
            args = server_cfg.get("args", [])
            env = server_cfg.get("env")
            cwd = server_cfg.get("cwd")
            if not isinstance(command, str) or not command:
                raise ValueError(f"MCP stdio server '{server_name}' requires 'command'")
            if not isinstance(args, list):
                raise ValueError(f"MCP stdio server '{server_name}' requires list 'args'")

            connections[server_name] = {
                "transport": "stdio",
                "command": command,
                "args": [str(a) for a in args],
                **({"env": env} if isinstance(env, dict) else {}),
                **({"cwd": cwd} if cwd is not None else {}),
            }
            continue

        url = server_cfg.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"MCP server '{server_name}' requires 'url'")

        headers = server_cfg.get("headers")
        conn: dict[str, Any] = {
            "transport": transport,
            "url": url,
        }
        if isinstance(headers, dict) and headers:
            conn["headers"] = headers

        connections[server_name] = conn

    if not connections:
        return None

    # We implement our own stable hashed prefixing in the registry layer.
    return MultiServerMCPClient(connections=connections, tool_name_prefix=False)
