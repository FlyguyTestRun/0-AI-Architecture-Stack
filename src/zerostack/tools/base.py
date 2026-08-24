"""Tool definitions shared by the built in tools and the MCP adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """The outcome of a tool call."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Tool:
    """A callable the agent may invoke.

    ``parameters`` is a JSON Schema object. It is the same shape MCP servers publish
    and the same shape function calling APIs expect, so a tool defined here can be
    handed to either without translation.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
    source: str = "builtin"

    def __call__(self, **kwargs: Any) -> ToolResult:
        try:
            return self.handler(**kwargs)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=f"{type(exc).__name__}: {exc}")

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
