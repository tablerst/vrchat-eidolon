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
   - `uv run vrchat-eidolon devices` (optional, to find audio device names)
   - `uv run vrchat-eidolon run`

## Milestone 1: Realtime Speech Loop

This milestone runs a minimal closed-loop:

Mic (pcm16) → Qwen-Omni-Realtime (WebSocket, server VAD) → `response.audio.delta` (pcm24) → local playback.

Defaults (configurable in `configs/app.yaml`):

- WebSocket URL: `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
- Model: `qwen3-omni-flash-realtime`
- Auth: `Authorization: Bearer ${DASHSCOPE_API_KEY}`

If you use VB-CABLE or a virtual device, set `audio.output.device` in `configs/dev.yaml`.

## Tests

- `uv run pytest`
