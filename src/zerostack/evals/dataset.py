"""The golden dataset: questions with known correct sources and answer phrases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    """One question and what a correct response looks like.

    ``expected_sources`` measures retrieval: did the right document come back.
    ``expected_phrases`` measures the answer: did the right fact reach the user.
    They are separate because a system can retrieve perfectly and still answer
    badly, and knowing which half broke is the whole point of measuring.
    """

    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_phrases: list[str] = field(default_factory=list)
    # Phrases that must not appear. Useful for pinning a fixed hallucination or
    # a leak that was closed, so it cannot come back unnoticed.
    forbidden_phrases: list[str] = field(default_factory=list)
    namespace: str = "default"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> EvalCase:
        return cls(
            question=str(payload["question"]),
            expected_sources=list(payload.get("expected_sources", [])),
            expected_phrases=list(payload.get("expected_phrases", [])),
            forbidden_phrases=list(payload.get("forbidden_phrases", [])),
            namespace=str(payload.get("namespace", "default")),
            tags=list(payload.get("tags", [])),
        )


@dataclass
class EvalDataset:
    """A named set of cases."""

    name: str
    cases: list[EvalCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def filter_by_tag(self, tag: str) -> EvalDataset:
        return EvalDataset(
            name=f"{self.name}[{tag}]",
            cases=[case for case in self.cases if tag in case.tags],
        )


def load_dataset(path: Path) -> EvalDataset:
    """Load a dataset from JSON.

    Shape: {"name": "...", "cases": [{"question": "...", "expected_sources": [...]}]}
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDataset(
        name=str(payload.get("name", Path(path).stem)),
        cases=[EvalCase.from_dict(case) for case in payload.get("cases", [])],
    )
