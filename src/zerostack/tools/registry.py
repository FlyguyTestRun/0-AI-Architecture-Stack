"""Tool registry (layer 5).

The registry is the single place the orchestrator looks for capabilities. Built in
tools and MCP discovered tools land in the same registry with the same shape, which
is the reason the orchestrator never needs to know MCP exists.
"""

from __future__ import annotations

import logging

from zerostack.observability import get_tracer
from zerostack.tools.base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """A name to tool mapping with traced invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning("tool %s is already registered, overwriting", tool.name)
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def call(self, name: str, **kwargs) -> ToolResult:
        """Invoke a tool by name, recording a span."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                content="",
                error=f"unknown tool '{name}'. Available: {', '.join(self.names())}",
            )
        with get_tracer().span("tool.call", tool=name, args=kwargs) as span:
            result = tool(**kwargs)
            span.set(ok=result.ok, error=result.error)
        return result
