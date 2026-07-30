# AI Platform

An AI platform with a **single MCP server** and a client-side agent that routes user queries to the right tools — or chains several tools in sequence.

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐
│ Web Client  │ ────────────► │  FastAPI Backend │
│ (agent UI)  │ ◄──────────── │  /chat /platform │
└─────────────┘               └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │ AgentOrchestrator│  ← OpenAI function calling
                              │  (ReAct loop)    │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │   MCPManager     │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │  MCP Server      │  math + web + file tools
                              │  mcp_servers/    │
                              │  server.py       │
                              └──────────────────┘
```

### Components

| Layer | Role |
|-------|------|
| `mcp_servers/server.py` | Single MCP server with all tools |
| `core/mcp_manager.py` | Connects to the MCP server and exposes tools to the agent |
| `agent/orchestrator.py` | LLM agent that picks one tool or a tool chain per query |
| `api/main.py` | REST API + serves the web client |
| `client/` | Browser UI showing answers and tool traces |

## Quick start

```powershell
cd ai-platform
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set OPENAI_API_KEY
python run.py
```

Open http://localhost:8000

## Example queries

- **Single tool:** `What is 144 divided by 12?` → routes to `divide`
- **Tool chain:** `Search for MCP protocol and write a summary to notes/mcp.txt` → `search_web` then `write_file`
- **Multi-step:** `Calculate 2^10 and save the result to result.txt` → `power` then `write_file`

## Add more tools

Edit `mcp_servers/server.py` and add new functions with `@mcp.tool()`. Restart the platform — tools appear automatically.

## API

### `GET /platform`
Returns the connected MCP server and its tools.

### `POST /chat`
```json
{ "message": "your question" }
```

Response:
```json
{
  "answer": "final natural language answer",
  "tool_calls": [
    { "tool": "multiply", "arguments": {"a": 15, "b": 23}, "result": "345" }
  ],
  "iterations": 2
}
```

## Configuration

- `config/mcp_servers.yaml` — MCP server config (in-memory or stdio transport)
- `.env` — `OPENAI_API_KEY`, optional `OPENAI_MODEL`, `API_PORT`

## Run MCP server standalone (stdio)

```powershell
python -m mcp_servers.server
```

Use this for Claude Desktop / Cursor MCP config.
