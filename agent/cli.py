"""Interactive CLI agent — routes queries to the remote MCP server."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from agent.orchestrator import AgentOrchestrator
from core.config import ROOT, get_mcp_server_url
from core.mcp_client import MCPClient

load_dotenv(ROOT / ".env")


async def main() -> None:
    client = MCPClient()
    try:
        tools = await client.list_tools()
    except Exception as exc:
        print(f"Could not connect to MCP server at {get_mcp_server_url()}: {exc}")
        print("Start it first with: python run_mcp.py")
        return

    print("AI Platform CLI Agent")
    print(f"MCP server: {client.url}")
    print("Connected MCP tools:", ", ".join(t.name for t in tools))
    print("Type 'exit' to quit.\n")

    orchestrator = await AgentOrchestrator.create()

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
