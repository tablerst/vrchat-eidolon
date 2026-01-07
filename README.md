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

### Audio sample rates (important)

If Realtime can hear you and returns text, but the audio you hear sounds like **garbled noise**, it is almost always a **sample-rate mismatch**.

The DashScope reference implementation for Omni Realtime uses:

- **Input:** 16 kHz, mono, PCM16
- **Output:** 24 kHz, mono (for `qwen3-omni-flash-realtime` output is `pcm24`)

Windows “Device Properties → Format” can show much higher rates (e.g. 96 kHz / 192 kHz). That is the *device mix format*, not necessarily the model wire format.

This repo will automatically resample between your device rates (`audio.input.sample_rate` / `audio.output.sample_rate`) and the model wire rates configured in:

- `qwen.realtime.input_sample_rate_hz` (default 16000)
- `qwen.realtime.output_sample_rate_hz` (default 24000)

If you still hear noise, try adjusting `qwen.realtime.output_sample_rate_hz` to match the model output you observe (common alternatives are 24000 or 48000).

## Tests

- `uv run pytest`
