# AGENTS.md

## Language policy (important)

- **All code comments MUST be written in English.**
- Developer-facing docs in this repository should be English-first.
- User-facing text (e.g., chat output) can be localized as needed, but keep internal naming/log fields in English.

Import policy (important):

- For in-repo modules, prefer fully-qualified imports under `vrchat_eidolon.*` (avoid implicit top-level imports).

---

## Architecture guardrails (read twice)

- **Dual-loop architecture** is intentional:
  - **Speech Loop**: low-latency audio I/O via Qwen-Omni-Realtime (WebSocket).
  - **Action Loop**: planning + tool execution (MCP) orchestrated at turn-level.
- **Do not put millisecond-level audio streaming inside LangGraph.** LangGraph is for turn-level control flow, fan-out tool execution, and state management.
- **Speak before act** is enforced by a latch/gate (no sleeps): do not execute “heavy” tools (move/turn/large gestures) until the first audio has actually started playing.
- **VRChat control is external**: this repo is an MCP *client* and orchestrator. Do not implement a VRChat OSC/MCP server inside this repository.

---

## Python code style & performance (read twice)

- Target Python version is **3.12+** (see `pyproject.toml`). Prefer modern stdlib idioms.
- Use **type hints** for public interfaces and cross-module boundaries.
- Prefer **small, explicit data objects** for events/state.
  - Use `@dataclass(slots=True)` for simple records (events, config structs) when it improves memory/perf.
  - Avoid overly clever metaprogramming; prioritize readability in hot paths.
- Be mindful of the **audio hot path**:
  - Avoid blocking calls and avoid excessive allocations/copies.
  - Prefer `memoryview`/bytes slices and explicit buffer boundaries.
- Logging:
  - Prefer structured logs.
  - Avoid `print()` in library code (tests/prototyping only).

---

## Concurrency & cancellation (read twice)

- Prefer **structured concurrency** for long-lived tasks:
  - Use `asyncio.TaskGroup` where possible.
  - Ensure clean cancellation and resource teardown (sockets, devices, sessions).
- **Never block the event loop** from audio callbacks:
  - Treat audio callbacks as real-time: do minimal work and hand off via a thread-safe queue.
  - Bridge thread → asyncio using `loop.call_soon_threadsafe(...)` (or equivalent).
- Backpressure must be explicit:
  - Use **bounded** queues.
  - Define drop strategy: typically **drop old frames** (vision) but avoid dropping audio unless explicitly acceptable.
- Tool fan-out must be bounded:
  - Enforce a concurrency cap (e.g., semaphore).
  - Tool execution should be cancelable, but must always clean up sessions/resources.
- Shared state:
  - Prefer message passing (events) over shared mutable state.
  - If using LangGraph state reducers, ensure merges are deterministic.

---

## Dev environment tips (avoid “guess-and-fix”)

---

## Use `uv` as the only package manager

- Python version: see `pyproject.toml` (currently `>=3.12`).
- Common commands:
  - Create venv: `uv venv --python 3.12`
  - Install/sync deps: `uv sync`
  - Run: `uv run python main.py`
  - Add dependency: `uv add <package>`
  - Add dev dependency: `uv add --dev <package>`

Notes:

- Keep `pyproject.toml` and `uv.lock` consistent (always use `uv add` / `uv sync`).

---

## Configuration strategy

- **YAML is the primary configuration**: put reproducible configuration in `configs/*.yaml`.
- **Environment variables are only for secrets and machine-specific overrides** (e.g., `DASHSCOPE_API_KEY`).
- The repository root `.env` is for local development injection only (do not commit it).

Sharp edges (read twice):

- `${ENV_VAR}` expansion is strict: missing/empty env should fail fast with a clear `ConfigError`.
- `qwen.api_key` may default from `DASHSCOPE_API_KEY`.

---

## Qwen model constraints (do not violate)

### Qwen-Omni (Chat Completions)

- **Must use `stream=True`**.
- **One user message may include text + only one other modality** (image OR audio OR video).
- **Thinking mode and audio output are incompatible**:
  - When `enable_thinking=true`, audio output is not supported.

Recommended two-phase design:

- **PLAN**: `modalities=["text"]` (focus on tool calls; deterministic, parseable output)
- **SPEAK**: `modalities=["text","audio"]` (final speech; speak-first UX)

### Qwen-Omni-Realtime (WebSocket)

- Use WebSocket Realtime for low-latency audio streaming.
- Prefer **server-side VAD**: `turn_detection.type=server_vad`.
- Output modalities are `['text']` or `['text','audio']`.
- Audio formats:
  - Input audio is **pcm16**.
  - `qwen3-omni-flash-realtime` output audio format is **pcm24**.
- Images:
  - JPG/JPEG only.
  - **≤ 500 KB before Base64 encoding**.
  - Recommended 480p/720p; **~1 FPS**.
  - **Must send at least one `input_audio_buffer.append` before `input_image_buffer.append`.**
  - Image buffer is submitted along with audio buffer on commit.
- A single Realtime session has a **maximum lifetime (~30 minutes)**. Implement session rotation / reconnect.

---
## Testing instructions

- Run tests:
  - `uv run pytest`

---

