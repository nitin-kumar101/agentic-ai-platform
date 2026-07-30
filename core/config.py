"""Platform configuration loader."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "mcp_servers.yaml"


@dataclass
class MCPServerConfig:
    id: str
    name: str
    description: str = ""
    transport: str = "in_memory"
    module: str | None = None
    attr: str = "mcp"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None


def load_server_configs(config_path: Path | None = None) -> list[MCPServerConfig]:
    path = config_path or DEFAULT_CONFIG
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    servers: list[MCPServerConfig] = []
    for entry in raw.get("servers", []):
        servers.append(
            MCPServerConfig(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                description=entry.get("description", ""),
                transport=entry.get("transport", "in_memory"),
                module=entry.get("module"),
                attr=entry.get("attr", "mcp"),
                command=entry.get("command"),
                args=entry.get("args", []),
                cwd=entry.get("cwd"),
                env=entry.get("env"),
            )
        )
    return servers


def load_server_config(config_path: Path | None = None) -> MCPServerConfig:
    """Load the single configured MCP server."""
    configs = load_server_configs(config_path)
    if not configs:
        raise ValueError("No MCP server configured in config/mcp_servers.yaml")
    if len(configs) > 1:
        raise ValueError("Only one MCP server is supported. Remove extra entries from config/mcp_servers.yaml")
    return configs[0]


def resolve_in_memory_server(config: MCPServerConfig) -> Any:
    if not config.module:
        raise ValueError(f"Server '{config.id}' requires a module for in_memory transport")
    module = importlib.import_module(config.module)
    return getattr(module, config.attr)


def get_openai_settings() -> dict[str, str]:
    return {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }
