"""MCP adapter (layer 5).

Model Context Protocol servers are the intended way to give the agent access to
GitHub, Slack, databases and the filesystem without writing a bespoke integration for
each one. This module discovers the tools a configured MCP server publishes and wraps
each one as a :class:`~zerostack.tools.base.Tool`, so the orchestrator sees no
difference between an MCP tool and a built in one.

Servers are declared in ``mcp.json`` at the repository root. The file uses the
conventional MCP client configuration shape, which most MCP capable clients read:

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
        }
      }
    }

Discovery requires the ``mcp`` package and the server command to be installed. When
either is missing this module logs and returns no tools, which leaves the rest of the
stack fully functional.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from zerostack.config import REPO_ROOT
from zerostack.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = REPO_ROOT / "mcp.json"
DISCOVERY_TIMEOUT_SECONDS = 20.0


def load_mcp_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the MCP server declarations. Returns an empty mapping when absent."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.debug("no mcp config at %s", config_path)
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not parse %s: %s", config_path, exc)
        return {}
    servers = payload.get("mcpServers", {})
    if not isinstance(servers, dict):
        logger.warning("mcpServers in %s is not an object", config_path)
        return {}
    return servers


async def _discover_async(server_name: str, spec: dict[str, Any]) -> list[Tool]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=spec["command"],
        args=spec.get("args", []),
        env=spec.get("env"),
    )

    tools: list[Tool] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            for remote in listing.tools:
                tools.append(
                    _wrap_remote_tool(
                        server_name,
                        spec,
                        remote.name,
                        remote.description or "",
                        remote.inputSchema or {"type": "object", "properties": {}},
                    )
                )
    return tools


def _wrap_remote_tool(
    server_name: str,
    spec: dict[str, Any],
    tool_name: str,
    description: str,
    schema: dict[str, Any],
) -> Tool:
    """Wrap one remote MCP tool as a synchronous local Tool."""

    def handler(**kwargs: Any) -> ToolResult:
        try:
            text = asyncio.run(_call_async(spec, tool_name, kwargs))
        except Exception as exc:
            return ToolResult(ok=False, content="", error=f"{type(exc).__name__}: {exc}")
        return ToolResult(ok=True, content=text, data={"server": server_name})

    return Tool(
        name=f"{server_name}__{tool_name}",
        description=description,
        parameters=schema,
        handler=handler,
        source=f"mcp:{server_name}",
    )


async def _call_async(spec: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=spec["command"], args=spec.get("args", []), env=spec.get("env")
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            parts = [getattr(block, "text", "") for block in getattr(result, "content", [])]
            return "\n".join(part for part in parts if part)


def discover_mcp_tools(config_path: Path | None = None) -> list[Tool]:
    """Discover tools from every configured MCP server.

    Failures are contained per server: one unreachable server does not prevent the
    others from registering their tools.
    """
    servers = load_mcp_config(config_path)
    if not servers:
        return []

    try:
        import mcp  # noqa: F401
    except ImportError:
        logger.info(
            "mcp.json declares %d server(s) but the 'mcp' package is not installed",
            len(servers),
        )
        return []

    discovered: list[Tool] = []
    for server_name, spec in servers.items():
        if not isinstance(spec, dict) or "command" not in spec:
            logger.warning("mcp server '%s' has no command, skipping", server_name)
            continue
        try:
            tools = asyncio.run(
                asyncio.wait_for(
                    _discover_async(server_name, spec), timeout=DISCOVERY_TIMEOUT_SECONDS
                )
            )
        except Exception as exc:
            logger.warning("mcp discovery failed for '%s': %s", server_name, exc)
            continue
        logger.info("mcp server '%s' published %d tool(s)", server_name, len(tools))
        discovered.extend(tools)

    return discovered
