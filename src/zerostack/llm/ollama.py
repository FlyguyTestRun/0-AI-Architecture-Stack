"""Ollama provider: the $0 local model server from layer 4 of the diagram."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from zerostack.llm.base import ChatMessage, LLMResponse

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Chat completions against a local Ollama server.

    Availability is checked against the /api/tags endpoint with a short timeout so
    that ``auto`` provider selection never blocks start up when Ollama is absent.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=1.5)
            response.raise_for_status()
        except Exception as exc:
            logger.debug("ollama unavailable at %s: %s", self.base_url, exc)
            return False
        return True

    def installed_models(self) -> list[str]:
        """Return the model tags the local server has pulled."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception:
            return []

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        response = httpx.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        body = response.json()
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=body.get("message", {}).get("content", "").strip(),
            model=payload["model"],
            provider=self.name,
            latency_ms=round(latency, 2),
            usage={
                "prompt_eval_count": body.get("prompt_eval_count"),
                "eval_count": body.get("eval_count"),
            },
        )
