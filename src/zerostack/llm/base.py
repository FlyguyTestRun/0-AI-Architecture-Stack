"""Protocol and value types shared by every LLM provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    """One message in a chat completion request."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """The result of a completion call."""

    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@runtime_checkable
class LLMClient(Protocol):
    """The interface every provider implements.

    Keeping this to two methods is deliberate. Anything richer (streaming, function
    calling) belongs behind a provider specific adapter so that swapping providers
    never breaks the orchestrator layer.
    """

    name: str
    model: str

    def available(self) -> bool:
        """Return True when this provider can serve a request right now."""
        ...

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        """Run a chat completion."""
        ...
