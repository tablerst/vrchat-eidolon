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
	- MCP wrapper: `src/tools/langchain_mcp_client.py`
	- Tool bridge + guardrails: `src/tools/mcp_bridge.py`, `src/tools/runtime.py`

Configuration supports both:

1) **Preferred (multi-server)**: `mcp.servers` (passed through to `MultiServerMCPClient`)
2) **Legacy (single server)**: `mcp.transport/url/command/args` (auto-translated into `mcp.servers.default`)

## Tests

- `uv run pytest`
