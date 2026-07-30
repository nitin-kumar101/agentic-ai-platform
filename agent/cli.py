"""Interactive CLI agent — routes queries to MCP tools without the web UI."""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from agent.orchestrator import AgentOrchestrator
from core.config import ROOT, load_server_config
from core.mcp_manager import MCPManager

load_dotenv(ROOT / ".env")


async def main() -> None:
    config = load_server_config()
    async with MCPManager(config) as manager:
        print("AI Platform CLI Agent")
        print("Connected MCP tools:", ", ".join(t.name for t in manager.tools))
        print("Type 'exit' to quit.\n")

        orchestrator = AgentOrchestrator(manager)

        while True:
            try:
                query = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                print("Bye.")
                break

            result = await orchestrator.run(query)
            print(f"\nAgent> {result.answer}\n")
            if result.tool_calls:
                print("Tool trace:")
                for step in result.tool_calls:
                    print(f"  - {step.tool}({step.arguments})")
                    preview = step.result[:200].replace("\n", " ")
                    print(f"    => {preview}{'...' if len(step.result) > 200 else ''}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
