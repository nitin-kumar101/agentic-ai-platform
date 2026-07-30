"""CLI entrypoint for the MCP server (Streamable HTTP)."""

import os

from dotenv import load_dotenv

from core.config import ROOT

load_dotenv(ROOT / ".env")

if __name__ == "__main__":
    from mcp_servers.server import mcp

    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8001"))

    print(f"Starting MCP server at http://{host}:{port}/mcp")
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=True,
    )
