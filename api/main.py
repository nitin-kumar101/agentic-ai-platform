"""FastAPI backend for the AI platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.orchestrator import AgentOrchestrator
from core.config import ROOT, load_server_config
from core.mcp_manager import MCPManager

load_dotenv(ROOT / ".env")

manager: MCPManager | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global manager
    config = load_server_config()
    manager = MCPManager(config)
    await manager.__aenter__()
    yield
    await manager.__aexit__(None, None, None)
    manager = None


app = FastAPI(
    title="AI Platform",
    description="AI platform with a single MCP server and tool-routing agent",
    lifespan=lifespan,
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
    return {"status": "ok"}


@app.get("/platform")
async def platform_info() -> dict[str, Any]:
    if manager is None:
        raise HTTPException(status_code=503, detail="Platform not ready")
    return {
        "server": manager.describe_platform(),
        "tool_count": len(manager.tools),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if manager is None:
        raise HTTPException(status_code=503, detail="Platform not ready")

    orchestrator = AgentOrchestrator(manager)
    result = await orchestrator.run(request.message)
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