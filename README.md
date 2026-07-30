# AI Platform

FastAPI agent + **separate MCP server** over Streamable HTTP.

## Architecture

```
Terminal 1                         Terminal 2
python run_mcp.py                  python run.py
     │                                  │
     ▼                                  ▼
MCP Server                      FastAPI (port 8000)
http://127.0.0.1:8001/mcp  ◄────  MCPClient (HTTP)
     │
  mcp_servers/server.py
  (all tools)
```

- **MCP server** runs on its own port (`8001`) in Streamable HTTP mode
- **FastAPI** runs on port `8000` and talks to the MCP server via `MCP_SERVER_URL`
- No in-process MCPManager — the agent connects over HTTP for each tool call

## Quick start

```powershell
cd ai-platform
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Set GROQ_API_KEY in .env
```

**Terminal 1 — start MCP server:**
```powershell
python run_mcp.py
```

**Terminal 2 — start FastAPI:**
```powershell
python run.py
```

Open http://localhost:8000

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Groq API key for the agent |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model |
| `MCP_HOST` | `127.0.0.1` | MCP server bind host |
| `MCP_PORT` | `8001` | MCP server port |
| `MCP_SERVER_URL` | `http://127.0.0.1:8001/mcp` | URL FastAPI uses to reach MCP |
| `API_PORT` | `8000` | FastAPI port |

## CLI agent

```powershell
python run_mcp.py          # terminal 1
python -m agent.cli        # terminal 2
```
