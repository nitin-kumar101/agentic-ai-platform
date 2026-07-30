"""LLM agent that selects and chains MCP tool calls based on user intent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from groq import Groq
from core.config import get_openai_settings
from langchain_groq import ChatGroq
from core.mcp_manager import MCPManager


SYSTEM_PROMPT = """You are an AI platform agent with access to one MCP server and its tools.

Your job:
1. Understand the user's request.
2. Choose the single best tool OR a short sequence of tools to satisfy it.
3. Call tools with valid JSON arguments matching each tool schema.
4. When a task needs multiple steps (e.g. search then write file, calculate then save), call tools in order and use prior outputs in later steps.
5. If the user's request is not clear, ask for clarification.
6. If tool is not able to perform the task because of invalid argument given by you then try to correct the argument and call the tool again.
7. After tools finish, reply with a concise, helpful final answer.

Rules:
- Prefer the minimum number of tool calls.
- Never invent tool names — only use provided functions.
- If no tool is needed, answer directly.
- When tool output is JSON, interpret it for the user instead of dumping raw JSON unless they asked for it.
"""


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


class AgentOrchestrator:
    """ReAct-style orchestrator over the unified MCP tool registry."""

    def __init__(self, manager: MCPManager, max_iterations: int = 8) -> None:
        self.manager = manager
        self.max_iterations = max_iterations
        # settings = get_openai_settings()
        # if not settings["api_key"]:
        #     raise ValueError("OPENAI_API_KEY is required. Copy .env.example to .env and set your key.")
        # self.client = OpenAI(api_key=settings["api_key"])
        self.groq_client = Groq(api_key="")
        self.model = "llama-3.1-8b-instant"
        self.openai_tools = manager.list_openai_tools()


    async def run(self, user_query: str) -> AgentResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        tool_history: list[ToolCallRecord] = []
        iterations = 0
        final_answer = ""

        while iterations < self.max_iterations:
            iterations += 1
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )

                for call in message.tool_calls:
                    args = json.loads(call.function.arguments or "{}")
                    result = await self.manager.call_tool(call.function.name, args)
                    tool_history.append(
                        ToolCallRecord(
                            tool=call.function.name,
                            arguments=args,
                            result=result,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
                continue

            final_answer = message.content or "Done."
            break

        if not final_answer and tool_history:
            final_answer = "Completed tool execution. See tool trace for details."

        return AgentResponse(
            answer=final_answer,
            tool_calls=tool_history,
            iterations=iterations,
        )
