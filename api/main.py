"""FastAPI backend for the AI platform."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.orchestrator import AgentOrchestrator
from core.config import ROOT, get_mcp_server_url
from core.mcp_client import MCPClient

load_dotenv(ROOT / ".env")

app = FastAPI(
    title="AI Platform",
    description="FastAPI agent that talks to a remote MCP server over Streamable HTTP",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query for the agent")


class ToolCallView(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: str


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallView]
    iterations: int


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mcp_server_url": get_mcp_server_url()}


@app.get("/platform")
async def platform_info() -> dict[str, Any]:
    client = MCPClient()
    try:
        server = await client.describe_platform()
        tools = await client.list_tools()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MCP server unavailable at {client.url}: {exc}",
        ) from exc
    return {"server": server, "tool_count": len(tools)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        orchestrator = await AgentOrchestrator.create()
        result = await orchestrator.run(request.message)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        tool_calls=[
            ToolCallView(tool=call.tool, arguments=call.arguments, result=call.result)
            for call in result.tool_calls
        ],
        iterations=result.iterations,
    )


client_dir = ROOT / "client"
app.mount("/static", StaticFiles(directory=client_dir), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(client_dir / "index.html")