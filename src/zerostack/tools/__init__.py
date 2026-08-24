"""Tool layer (layer 5)."""

from zerostack.tools.base import Tool, ToolResult
from zerostack.tools.builtin import build_builtin_tools
from zerostack.tools.mcp import discover_mcp_tools, load_mcp_config
from zerostack.tools.registry import ToolRegistry


def build_registry(include_mcp: bool = True) -> ToolRegistry:
    """Build the registry the orchestrator uses."""
    registry = ToolRegistry()
    registry.register_many(build_builtin_tools())
    if include_mcp:
        registry.register_many(discover_mcp_tools())
    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "build_builtin_tools",
    "build_registry",
    "discover_mcp_tools",
    "load_mcp_config",
]
