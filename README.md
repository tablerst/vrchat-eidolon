# VRChat Eidolon

VRChat Eidolon is a **speak-first, act-after** embodied agent orchestrator for VRChat.

- This repository is an **MCP client** and orchestrator.
- VRChat control is external (provided by an MCP server). This repo does **not** implement a VRChat OSC/MCP server.

## Development

- Python: 3.12+
- Package manager: `uv`

## Quickstart

1. Copy `.env.example` to `.env` and fill in `DASHSCOPE_API_KEY`.
2. Run:
   - `uv run vrchat-eidolon --help`
   - `uv run vrchat-eidolon run`

## Tests

- `uv run pytest`
