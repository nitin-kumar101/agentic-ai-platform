"""Connect to a remote MCP server over Streamable HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.types import TextContent

from core.config import get_mcp_server_url

try:
    from mcp import Client as MCPHttpClient
except ImportError:
    MCPHttpClient = None  # type: ignore[misc, assignment]


@dataclass
class RegisteredTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]


@asynccontextmanager
async def _mcp_session(url: str) -> AsyncIterator[Any]:
    """Open an MCP session for mcp 2.x (`Client`) or 1.x (streamable HTTP)."""
    if MCPHttpClient is not None:
        async with MCPHttpClient(url) as session:
            yield session
        return

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


class MCPClient:
    """Stateless client for an MCP server running on Streamable HTTP."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or get_mcp_server_url()

    async def list_tools(self) -> list[RegisteredTool]:
        async with _mcp_session(self.url) as session:
            result = await session.list_tools()
            return [
                RegisteredTool(
                    name=tool.name,
                    title=getattr(tool, "title", None),
                    description=tool.description,
                    input_schema=tool.input_schema,
                )
                for tool in result.tools
            ]

    def list_openai_tools(self, tools: list[RegisteredTool]) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            description = tool.description or tool.title or tool.name
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return openai_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        async with _mcp_session(self.url) as session:
            result = await session.call_tool(tool_name, arguments or {})

        if getattr(result, "is_error", False):
            return self._content_to_text(result.content)

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return json.dumps(structured, indent=2)

        return self._content_to_text(result.content)

    @staticmethod
    def _content_to_text(content: list[Any]) -> str:
        chunks: list[str] = []
        for block in content:
            if isinstance(block, TextContent):
                chunks.append(block.text)
            else:
                chunks.append(str(block))
        return "\n".join(chunks) if chunks else ""

    async def describe_platform(self) -> dict[str, str]:
        tools = await self.list_tools()
        return {
            "server_id": "platform",
            "name": "AI Platform MCP Server",
            "description": f"Streamable HTTP MCP server at {self.url}",
            "tools": ", ".join(tool.name for tool in tools),
        }
