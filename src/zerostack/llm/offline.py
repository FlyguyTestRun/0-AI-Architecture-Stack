"""An offline provider that needs no model server.

This exists so that a clean clone is runnable and so that CI can exercise the whole
pipeline deterministically. It is not a language model. It performs extractive
question answering: it scores the sentences in the supplied context against the
question and returns the best ones.

That behaviour is chosen on purpose. An echo provider would make the demo look
broken, while an extractive provider produces a genuinely useful answer whenever the
retrieval layer did its job, which is exactly the property you want to test.
"""

from __future__ import annotations

import re
import time
from typing import Any

from zerostack.llm.base import ChatMessage, LLMResponse

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


# The orchestrator tags context and tool messages with these prefixes. They are the
# contract between generate_node and this provider.
CONTEXT_PREFIX = "Context:"
TOOL_PREFIX = "Tool result:"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


_CITATION_HEADER_RE = re.compile(r"^\[\d+\]\s+source:\s+\S+\s*$", re.MULTILINE)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
# Scaffolding the graph tier adds around its evidence. These lines label the
# context; they are not part of it. Scoring them let the labels outrank the
# sentences they introduce, which is the same defect citation headers caused.
_GRAPH_LABEL_RE = re.compile(r"^(Relationships across documents:|Related to:.*)$", re.MULTILINE)
# "- subject and object [source.md]: evidence sentence" keeps the evidence and
# drops the triple, which is machine syntax rather than something to quote.
_GRAPH_RELATION_PREFIX_RE = re.compile(r"^-\s+.*?\s+\[[^\]]+\]:\s*", re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    """Split context into scoreable sentences.

    Citation headers and markdown headings are removed first. They are structural
    scaffolding, not content, and leaving them in lets a heading win on term overlap
    while carrying no information.
    """
    cleaned = _CITATION_HEADER_RE.sub("", text)
    cleaned = _MARKDOWN_HEADING_RE.sub("", cleaned)
    cleaned = _GRAPH_LABEL_RE.sub("", cleaned)
    cleaned = _GRAPH_RELATION_PREFIX_RE.sub("", cleaned)
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", cleaned)
    return [p.strip() for p in parts if len(p.strip()) > 20]


class OfflineLLM:
    """Deterministic extractive provider. Always available."""

    name = "offline"

    def __init__(self, model: str = "offline-extractive", max_sentences: int = 4) -> None:
        self.model = model
        self.max_sentences = max_sentences

    def available(self) -> bool:
        return True

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        start = time.perf_counter()
        question = ""
        for message in reversed(messages):
            if message.role == "user":
                question = message.content
                break

        # Only messages the orchestrator explicitly tagged as context or tool output
        # are treated as source material. The instruction prompt is not content, and
        # scoring it against the question would leak the rules into the answer.
        context = "\n\n".join(
            m.content.removeprefix(CONTEXT_PREFIX).strip()
            for m in messages
            if m.role == "system" and m.content.startswith(CONTEXT_PREFIX)
        )
        tool_output = "\n".join(
            m.content.removeprefix(TOOL_PREFIX).strip()
            for m in messages
            if m.role == "system" and m.content.startswith(TOOL_PREFIX)
        ).strip()

        text = self._answer(question, context, tool_output)
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=round(latency, 2),
            usage={
                "mode": "extractive",
                "context_chars": len(context),
                "used_tool_output": bool(tool_output),
            },
        )

    def _answer(self, question: str, context: str, tool_output: str = "") -> str:
        # A deterministic tool result outranks extraction: it is a computed fact.
        if tool_output and not context.strip():
            return tool_output
        if not context.strip():
            return (
                "No context was retrieved for this question, and no language model is "
                "running. Start Ollama with `make up` for generated answers, or ingest "
                "documents with `zerostack ingest` so the retrieval layer has something "
                "to work with."
            )

        query_terms = set(_tokenize(question))
        # A question with only stopwords cannot be scored, so return the opening context.
        if not query_terms:
            sentences = _split_sentences(context)[: self.max_sentences]
            return " ".join(sentences)

        scored: list[tuple[float, int, str]] = []
        # The same sentence can reach the context twice, once as a retrieved
        # passage and once as graph evidence. Without this the answer repeats
        # itself verbatim, which reads like a defect even though both copies are
        # legitimately present in the input.
        seen: set[str] = set()
        for index, sentence in enumerate(_split_sentences(context)):
            fingerprint = " ".join(sentence.lower().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            terms = _tokenize(sentence)
            if not terms:
                continue
            overlap = sum(1 for t in terms if t in query_terms)
            if overlap == 0:
                continue
            # Normalise by length so a long paragraph does not win on volume alone.
            score = overlap / (len(terms) ** 0.5)
            scored.append((score, index, sentence))

        if not scored:
            if tool_output:
                return tool_output
            return (
                "The retrieved context does not appear to answer that question. "
                "Try rephrasing, or ingest more source documents."
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[: self.max_sentences]
        # Restore original document order so the answer reads naturally.
        top.sort(key=lambda item: item[1])
        extracted = " ".join(sentence for _, _, sentence in top)
        return f"{tool_output}\n\n{extracted}" if tool_output else extracted
