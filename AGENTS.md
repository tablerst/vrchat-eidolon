# AGENTS.md

## Language policy (important)

- **All code comments MUST be written in English.**
- Developer-facing docs in this repository should be English-first.
- User-facing text (e.g., chat output) can be localized as needed, but keep internal naming/log fields in English.

Import policy (important):

- For in-repo modules, prefer fully-qualified imports under `vrchat_eidolon.*` (avoid implicit top-level imports).

---

## Dev environment tips (avoid “guess-and-fix”)

### The “Context Triple”: entrypoint → adapters → guards

When debugging or adding features, always locate and reason about this chain:

- **Entrypoint**: CLI / `main.py` / session orchestration (LISTEN → PLAN → ACT → SPEAK → UPDATE)
- **Adapters**: Qwen OpenAI-compat client, streaming tool-call accumulator, audio I/O, MCP client-side runtime
- **Guards**: tool governance (allow-all by default), rate limiting, argument validation/clamping, timeouts/retries, and “external service not available” handling

Implementation anchors (start here, then branch out):

- Orchestration: `vrchat_eidolon/runtime/lifecycle.py`, `vrchat_eidolon/graph/build.py`, `vrchat_eidolon/graph/state.py`, `vrchat_eidolon/graph/nodes/*`
- Qwen adapter: `vrchat_eidolon/llm/client.py` (and its streamed tool-call accumulation)
- Tool governance: `vrchat_eidolon/mcp/policy.py`, `vrchat_eidolon/mcp/registry.py`
- Config: `vrchat_eidolon/config.py`
- Observability: `vrchat_eidolon/runtime/logging.py`

> Note: **VRChat OSC is NOT implemented in this repo anymore.**
> VRChat-related control is provided by an **external MCP service**. This repo focuses on orchestration, safety/guards, and audio.

### Write 3 verifiable assertions before changing code

Before making any fixes, write down (or encode as tests) at least 3 assertions:

1. Expected behavior (happy path)
2. Boundary case(s)
3. Failure mode(s) and the desired recovery behavior

### Observe first, tune later

On the critical path, ensure logs/metrics include at least:

- `trace_id`, `session_id`, `turn_id`
- `state` (PLAN/ACT/SPEAK…)
- latency (model latency, tool latency)
- tool execution outcomes

---

## Use `uv` as the only package manager

- Python version: see `pyproject.toml` (currently `>=3.12`).
- Common commands:
  - Create venv: `uv venv --python 3.12`
  - Install/sync deps: `uv sync`
  - Run: `uv run python main.py`
  - Add dependency: `uv add <package>`
  - Add dev dependency: `uv add --dev <package>`

---

## Configuration strategy

- **YAML is the primary configuration**: put reproducible configuration in `configs/*.yaml`.
- **Environment variables are only for secrets and machine-specific overrides** (e.g., `DASHSCOPE_API_KEY`).
- The repository root `.env` is for local development injection only (do not commit it).

Sharp edges (read twice):

- `${ENV_VAR}` expansion is strict: missing/empty env should fail fast with a clear `ConfigError`.
- `qwen.api_key` may default from `DASHSCOPE_API_KEY`.
- Prefer explicit tool `whitelist` in configs when you want to restrict tools; an empty whitelist is effectively “allow all”.

---

## Qwen-Omni hard constraints (do not violate)

- **Must use `stream=True`**.
- **One request may include text + only one other modality**.
- Recommended two-phase design:
  - **PLAN**: `modalities=["text"]` (focus on tool calls)
  - **SPEAK**: `modalities=["text","audio"]` (final speech)

Practical notes:

- `vrchat_eidolon/llm/client.py` should enforce `stream=True` for all Omni calls.
- If you see “Missing dependency 'openai'”, fix the environment (not the code): install it via `uv add openai`.

---

## Streaming tool-call accumulation

Tool-call `arguments` may arrive split across multiple streaming chunks.

- Do **not** parse half JSON.
- Accumulate per `tool_call_id` until the JSON becomes parseable.
- Log useful debug context:
  - accumulated length per tool_call_id
  - last JSON decode error (message + position)

Rule of thumb: treat tool-call JSON as a stream, not a value.

If you change tool calling, keep these invariants:

- Accumulate by `tool_call_id`.
- Probe parseability for logging only.
- Only execute tools from fully-parsed dict arguments.

---

## Tools: safety defaults

Local tool runtime is the first safety boundary.

- Default posture: **allow-all by default** when `tools.whitelist` is empty.
- Hard safety floor:
  - Global kill switch: `tools.enabled=false`
  - Per-tool rate limits (recommended for chatbox / high-frequency tools)
  - Optional argument validation/clamping
- Always return a structured `ToolResult` (ok/content/error) and keep failures non-fatal.
- Add per-turn caps (`tools.max_calls_per_turn`) at the orchestrator layer to prevent runaway loops.

Tool naming note:

- Model-facing names must be OpenAI-safe (letters/digits/underscore/hyphen).
- MCP tool names are unified and OpenAI-compatible (e.g. `vrc_chat_send`).

---

## Windows audio pitfalls (common failures)

- If using VB-CABLE: keep everything at **48kHz** (device sample-rate mismatches often cause pitch shifts, crackling, or stutter).
- If audio is silent:
  - verify the output device selected by the app
  - verify the input device selected by VRChat (or the target app)
  - check exclusive-mode settings in Windows sound control panel

---

## LangGraph (optional but recommended for scale)

Consider introducing LangGraph when:

- you need controlled tool loops (retry/clarify/degrade)
- you want explicit branching and error recovery
- orchestration complexity is growing beyond a simple state machine

Keep adoption incremental:

- start with a small graph that mirrors PLAN → ACT → SPEAK
- keep observability fields stable (`trace_id/session_id/turn_id/state`)
- gate it behind a config flag to avoid breaking existing flows

Non-negotiable: keep observability fields stable (`trace_id/session_id/turn_id/state`) across refactors.

---

## Testing instructions

- Run tests:
  - `uv run pytest`

Tip: prioritize unit tests for streaming accumulation, tool guards (allowlist/rate limit), and orchestration edge cases.

---

## PR instructions

### Commit message convention

Commit messages must follow Angular convention: `type(scope): subject`.

Suggested scopes:

- `orchestrator` / `tools` / `audio` / `qwen` / `mcp` / `langgraph` / `docs`

Examples:

- `fix(qwen): accumulate streamed tool_call arguments safely`
- `feat(tools): add allowlist and per-tool rate limit`
- `feat(langgraph): add graph-based orchestrator behind flag`
- `docs(plan): remove OSC implementation steps`

### Keep PRs small and reviewable

- One intent per PR.
- The PR description should answer: what changed, why, and how it was verified (command + key scenarios).