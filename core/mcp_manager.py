"""Connect to the MCP server and expose its tools to the agent."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from core.config import MCPServerConfig, resolve_in_memory_server


@dataclass
class RegisteredTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]


class MCPManager:
    """Lifecycle manager for the configured MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._stack = AsyncExitStack()
        self._client: Client | None = None
        self.tools: list[RegisteredTool] = []

    async def __aenter__(self) -> "MCPManager":
        self._client = await self._connect_server(self.config)
        result = await self._client.list_tools()
        self.tools = [
            RegisteredTool(
                name=tool.name,
                title=tool.title,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stack.aclose()
        self._client = None
        self.tools.clear()

    async def _connect_server(self, config: MCPServerConfig) -> Client:
        if config.transport == "in_memory":
            server = resolve_in_memory_server(config)
            client = Client(server)
        elif config.transport == "stdio":
            if not config.command:
                raise ValueError(f"Server '{config.id}' requires command for stdio transport")
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            )
            transport = await self._stack.enter_async_context(stdio_client(params))
            client = Client(transport)
        else:
            raise ValueError(f"Unsupported transport: {config.transport}")

        return await self._stack.enter_async_context(client)

    def list_openai_tools(self) -> list[dict[str, Any]]:
        """Convert MCP tools into OpenAI function-calling schema."""
        openai_tools: list[dict[str, Any]] = []
        for tool in self.tools:
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
        if self._client is None:
            raise RuntimeError("MCP server is not connected")

        result = await self._client.call_tool(tool_name, arguments or {})
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

    def describe_platform(self) -> dict[str, str]:
        return {
            "server_id": self.config.id,
            "name": self.config.name,
            "description": self.config.description,
            "tools": ", ".join(tool.name for tool in self.tools),
        }
