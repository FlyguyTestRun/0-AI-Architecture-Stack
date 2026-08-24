"""Document chunking.

A paragraph aware splitter with character overlap. It keeps paragraphs intact when
they fit, which preserves more meaning than a fixed window and costs nothing extra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable unit of text plus its provenance."""

    text: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.source}::{self.index}"


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(
                Chunk(
                    text=buffer.strip(),
                    source=source,
                    index=len(chunks),
                    metadata=dict(metadata or {}),
                )
            )
        buffer = ""

    for paragraph in paragraphs:
        # A single oversized paragraph is windowed rather than dropped.
        if len(paragraph) > chunk_size:
            flush()
            step = chunk_size - chunk_overlap
            for start in range(0, len(paragraph), step):
                window = paragraph[start : start + chunk_size]
                if window.strip():
                    chunks.append(
                        Chunk(
                            text=window.strip(),
                            source=source,
                            index=len(chunks),
                            metadata=dict(metadata or {}),
                        )
                    )
            continue

        if len(buffer) + len(paragraph) + 2 > chunk_size:
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            flush()
            buffer = f"{tail}\n\n{paragraph}" if tail else paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph

    flush()
    return chunks
