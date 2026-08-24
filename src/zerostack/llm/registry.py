"""Provider selection for the LLM layer."""

from __future__ import annotations

import logging

from zerostack.config import LLMSettings, get_settings
from zerostack.llm.base import LLMClient
from zerostack.llm.offline import OfflineLLM
from zerostack.llm.ollama import OllamaLLM

logger = logging.getLogger(__name__)


def build_llm(settings: LLMSettings | None = None) -> LLMClient:
    """Return an LLM client according to configuration.

    ``auto`` prefers Ollama when it is reachable and falls back to the offline
    extractive provider otherwise. Explicit provider names are honoured even when the
    backing service is missing, so that a misconfigured production deployment fails
    loudly instead of silently degrading.
    """
    settings = settings or get_settings().llm

    if settings.provider == "echo":
        return OfflineLLM()

    ollama = OllamaLLM(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout_seconds=settings.timeout_seconds,
    )

    if settings.provider == "ollama":
        return ollama

    if ollama.available():
        logger.info("llm layer: using ollama model %s", settings.model)
        return ollama

    logger.info("llm layer: ollama not reachable, using offline extractive provider")
    return OfflineLLM()
