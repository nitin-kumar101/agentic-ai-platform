"""Connect to a remote MCP server over Streamable HTTP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp import Client
from mcp.types import TextContent

from core.config import get_mcp_server_url


@dataclass
class RegisteredTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]


class MCPClient:
    """Stateless client for an MCP server running on Streamable HTTP."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or get_mcp_server_url()

    async def list_tools(self) -> list[RegisteredTool]:
        async with Client(self.url) as client:
            result = await client.list_tools()
            return [
                RegisteredTool(
                    name=tool.name,
                    title=tool.title,
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
        async with Client(self.url) as client:
            result = await client.call_tool(tool_name, arguments or {})

        if result.is_error:
            return self._content_to_text(result.content)

        if result.structured_content is not None:
            return json.dumps(result.structured_content, indent=2)

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
