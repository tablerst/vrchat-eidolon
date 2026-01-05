from __future__ import annotations

import argparse
import os

from observability.logging import configure_logging, get_logger
from orchestrator.simple import Orchestrator
from qwen.client import FakeQwenClient, QwenClient
from tools.mcp_bridge import register_mcp_tools
from tools.langchain_mcp_client import LangchainMcpClient, LangchainMcpClientConfig
from tools.openai_tools import get_openai_tool_specs
from tools.runtime import ToolRegistry

from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VRChat Eidolon")
    p.add_argument("--config", default="configs/app.yaml", help="YAML 配置路径")
    p.add_argument("--log-level", default="INFO", help="日志级别")
    p.add_argument("--text", default="你好", help="MVP：文本输入")
    p.add_argument("--fake", action="store_true", help="使用 FakeQwenClient（离线 stub）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(level=args.log_level)
    log = get_logger("eidolon.cli", level=args.log_level)

    # Offline stub: allow running without a real key.
    if args.fake and not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = "k_fake"

    cfg = load_config(args.config)

    tools = ToolRegistry(
        enabled=cfg.tools.enabled,
        whitelist=cfg.tools.whitelist,
        rate_limits={name: float(cps) for name, cps in cfg.tools.rate_limit.items()},
    )

    # Built-in echo tool for local debugging.
    tools.register("echo", lambda a: {"echo": a})

    plan_tools: list[dict[str, object]] | None = None
    if cfg.mcp.enabled:
        # MCP connections/tools are managed via `langchain-mcp-adapters`.
        # `cfg.mcp.servers` is the preferred shape; legacy config is translated by load_config.
        mcp = LangchainMcpClient(LangchainMcpClientConfig(servers=dict(cfg.mcp.servers)))

        # MVP-0: expose a small set of MCP-backed tools.
        exposed = ["vrc_status", "vrc_chat_send"]
        register_mcp_tools(tools, mcp=mcp, tool_names=exposed)
        plan_tools = get_openai_tool_specs(exposed)

    if args.fake:
        qwen = FakeQwenClient()
    else:
        qwen = QwenClient(cfg.qwen)

    orch = Orchestrator(
        qwen=qwen,
        tools=tools,
        plan_tools=plan_tools,
        max_calls_per_turn=cfg.tools.max_calls_per_turn,
    )
    out = orch.run_turn_text(args.text)

    log.info(
        "turn_output",
        assistant_text=out.assistant_text,
        tool_results=[r.__dict__ for r in out.tool_results],
    )
    return 0
