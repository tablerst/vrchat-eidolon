# vrchat-eidolon

MVP goal: follow the state machine described in `PLAN.md` (LISTEN → PLAN → ACT → SPEAK → UPDATE).

（中文：MVP 目标是按 `PLAN.md` 的状态机跑通“语音输入 → 工具调用 → 输出”的闭环。）

## Development

- Python: >= 3.12
- Package manager: **uv only**

## Quickstart

1. Put `DASHSCOPE_API_KEY` in the repo root `.env` (do not commit it)
2. Sync dependencies: `uv sync`
3. Run (preferred): `uv run vrchat-eidolon --config configs/app.yaml`

Notes:
- `qwen.base_url` defaults to the domestic DashScope OpenAI-compatible endpoint in `configs/*.yaml`.
	You typically do **not** need `DASHSCOPE_BASE_URL` unless you want to override it.

Optional (without relying on the installed console script): `uv run python main.py --config configs/app.yaml`

## MCP integration

This repo uses **`langchain-mcp-adapters`** to manage MCP connections and load MCP tools.

- Code path:
	- Runtime lifecycle (outer loop): `vrchat_eidolon/runtime/lifecycle.py`
	- LangGraph tick graph: `vrchat_eidolon/graph/build.py`, `vrchat_eidolon/graph/state.py`, `vrchat_eidolon/graph/nodes/*`
	- MCP client + registry + policy: `vrchat_eidolon/mcp/client.py`, `vrchat_eidolon/mcp/registry.py`, `vrchat_eidolon/mcp/policy.py`
	- LLM wrapper (OpenAI-compatible): `vrchat_eidolon/llm/client.py`

Configuration:

1) **Multi-server**: `mcp.servers` (passed through to `MultiServerMCPClient`)
2) **Single-server**: still uses `mcp.servers` with exactly one server

Tool naming:

- If there is **one** MCP server, tools keep their original names.
- If there are **multiple** MCP servers, tools are exposed to the model as:
  - `{prefix}__{toolName}`
  - `prefix` is a stable 4-character hash derived from `server_key`, and the first char is guaranteed to be non-numeric.

Note: this naming policy is documented in `PLAN.md`. The implementation lives under `vrchat_eidolon/mcp/`.

## Tests

- `uv run pytest`
