"""LLM agent that selects and chains MCP tool calls based on user intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from pydantic import create_model

from deepagents import create_deep_agent

from core.config import get_groq_settings
from core.mcp_client import MCPClient, RegisteredTool


INSTRUCTIONS = """You are an AI platform agent with access to one MCP server and its tools.
Your job:
1. Understand the user's request.
2. Choose the single best tool OR a short sequence of tools to satisfy it.
3. Call tools with valid arguments matching each tool's schema.
4. When a task needs multiple steps (e.g. search then write file, calculate then save),
   call tools in order and use prior outputs in later steps.
5. If the user's request is not clear, ask for clarification.
6. If a tool call fails because of an invalid argument you supplied, correct the
   argument and try again.
7. After tools finish, reply with a concise, helpful final answer.

Rules:
- Prefer the minimum number of tool calls.
- Never invent tool names — only use provided tools.
- If no tool is needed, answer directly.
- When tool output is JSON, interpret it for the user instead of dumping raw JSON
  unless they asked for it.
"""

_JSON_SCHEMA_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0


def _pydantic_model_from_json_schema(name: str, schema: dict[str, Any]):
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_SCHEMA_TYPES.get(prop_schema.get("type"), Any)
        default = ... if prop_name in required else prop_schema.get("default", None)
        fields[prop_name] = (py_type, default)

    return create_model(f"{name}_Args", **fields)  # type: ignore[call-overload]


def _build_langchain_tools(client: MCPClient, tools: list[RegisteredTool]) -> list[StructuredTool]:
    langchain_tools: list[StructuredTool] = []

    for entry in client.list_openai_tools(tools):
        fn = entry["function"]
        tool_name = fn["name"]
        description = fn.get("description", "")
        parameters = fn.get("parameters", {"type": "object", "properties": {}})
        args_schema = _pydantic_model_from_json_schema(tool_name, parameters)

        async def _call(tool_name: str = tool_name, **kwargs: Any) -> str:
            return await client.call_tool(tool_name, kwargs)

        langchain_tools.append(
            StructuredTool.from_function(
                name=tool_name,
                description=description,
                args_schema=args_schema,
                coroutine=_call,
            )
        )

    return langchain_tools


class AgentOrchestrator:
    """Deep-agent orchestrator that calls a remote MCP server over HTTP."""

    def __init__(
        self,
        client: MCPClient,
        tools: list[StructuredTool],
        max_iterations: int = 8,
    ) -> None:
        self.client = client
        self.max_iterations = max_iterations
        settings = get_groq_settings()
        if not settings["api_key"]:
            raise ValueError("GROQ_API_KEY is required. Copy .env.example to .env and set your key.")

        self.model = ChatGroq(model=settings["model"], api_key="")
        self.tools = tools
        self.agent = create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=INSTRUCTIONS,
        )

    @classmethod
    async def create(cls, max_iterations: int = 8) -> "AgentOrchestrator":
        client = MCPClient()
        registered_tools = await client.list_tools()
        langchain_tools = _build_langchain_tools(client, registered_tools)
        return cls(client=client, tools=langchain_tools, max_iterations=max_iterations)

    async def run(self, user_query: str) -> AgentResponse:
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"recursion_limit": self.max_iterations * 2},
        )

        messages = result.get("messages", [])
        tool_history: list[ToolCallRecord] = []
        final_answer = ""

        pending_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        for msg in messages:
            msg_type = getattr(msg, "type", None)

            if msg_type == "ai":
                for call in getattr(msg, "tool_calls", None) or []:
                    pending_calls[call["id"]] = (call["name"], call.get("args", {}))
                if getattr(msg, "content", None):
                    final_answer = msg.content

            elif msg_type == "tool":
                call_id = getattr(msg, "tool_call_id", None)
                name, args = pending_calls.pop(call_id, (getattr(msg, "name", "unknown"), {}))
                tool_history.append(
                    ToolCallRecord(tool=name, arguments=args, result=str(msg.content))
                )

        if not final_answer and tool_history:
            final_answer = "Completed tool execution. See tool trace for details."

        return AgentResponse(
            answer=final_answer,
            tool_calls=tool_history,
            iterations=len(messages),
        )