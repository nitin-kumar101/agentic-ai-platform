"""Platform configuration."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_mcp_server_url() -> str:
    return os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")


def get_groq_settings() -> dict[str, str]:
    return {
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    }
