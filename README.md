# vrchat-eidolon

MVP goal: follow the state machine described in `PLAN.md` (LISTEN → PLAN → ACT → SPEAK → UPDATE), and strictly obey `plans/00-shared/01-shared-constraints.md`.

（中文：MVP 目标是按 `PLAN.md` 的状态机跑通“语音输入 → 工具调用 → 语音输出”的闭环，并严格遵守共享约束。）

## Development

- Python: >= 3.12
- Package manager: **uv only**

## Quickstart

1. Put `DASHSCOPE_API_KEY` in the repo root `.env` (do not commit it)
2. Sync dependencies: `uv sync`
3. Run (preferred): `uv run vrchat-eidolon --config configs/app.yaml`

Optional (without relying on the installed console script): `uv run python main.py --config configs/app.yaml`

## MCP integration

This repo uses **`langchain-mcp-adapters`** to manage MCP connections and load MCP tools.

- Code path:
	- LangGraph orchestrator: `src/orchestrator/graph_orchestrator.py`
	- MCP gateway + local guardrails: `src/tools/mcp_gateway.py`, `src/tools/runtime.py`
	- Multi-server tool naming: `src/tools/mcp_naming.py`
	- Dynamic OpenAI tool specs: `src/tools/mcp_specs.py`
	- ToolResult payload + tool messages: `src/tools/tool_result_codec.py`, `src/tools/tool_messages.py`

Configuration:

1) **Multi-server**: `mcp.servers` (passed through to `MultiServerMCPClient`)
2) **Single-server**: still uses `mcp.servers` with exactly one server

Tool naming:

- If there is **one** MCP server, tools keep their original names.
- If there are **multiple** MCP servers, tools are exposed to the model as:
  - `{prefix}__{toolName}`
  - `prefix` is a stable 4-character hash derived from `server_key`, and the first char is guaranteed to be non-numeric.

## Tests

- `uv run pytest`
