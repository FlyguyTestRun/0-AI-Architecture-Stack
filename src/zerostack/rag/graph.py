"""Graph RAG: retrieval over a knowledge graph built from the corpus.

Vector and keyword retrieval both answer the same shape of question: "which
passage looks like this query". They cannot answer questions whose answer is
spread across documents that never mention each other, because no single passage
contains it. "Which policies does the support lead have authority over" needs the
support lead connected to each policy that names them, and that connection exists
only across documents.

A knowledge graph makes those connections explicit. Entities become nodes, the
statements that mention them together become edges, and a query walks outward
from the entities it names to collect the neighbourhood that surrounds them.

Extraction here is deterministic rather than model driven. That is a deliberate
trade. A language model extracts richer relations, but it costs money per
document, is nondeterministic, and cannot run on the offline path, which would
put the whole graph tier behind a model server. Deterministic extraction is
weaker per document and free, testable and always available. Where a model is
available the extractor can be swapped through the same protocol.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# A capitalised run: "Support Lead", "Northwind Supply", "Severity 1".
_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+(?:[A-Z][a-z]{2,}|\d+)){0,3})\b")
# Identifiers a proper noun pattern misses: E-4471, ISO 27001, SOC 2, GDPR.
_IDENTIFIER = re.compile(r"\b(?:[A-Z]{2,}[- ]?\d{1,6}|[A-Z]{3,})\b")
# Sentence boundary, used to scope which entities count as co-occurring.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Openers and headings that a capitalised run would otherwise promote to entities.
_NOT_ENTITIES = frozenset(
    """the this that these those every each any all some when where which what who
    why how support agents staff error code policy section note however therefore
    for from with without under over into onto about during after before""".split()
)


_ARTICLES = ("the ", "a ", "an ")


def _strip_article(value: str) -> str:
    """Drop a leading article.

    Without this, "The Support Lead" and "Support Lead" become two nodes that
    never connect, splitting one entity's neighbourhood in half.
    """
    lowered = value.lower()
    for article in _ARTICLES:
        if lowered.startswith(article):
            return value[len(article) :].strip()
    return value


@dataclass(frozen=True)
class Entity:
    """A node in the graph."""

    name: str
    kind: str = "term"

    @property
    def key(self) -> str:
        return self.name.lower()


@dataclass
class Relation:
    """An edge, weighted by how often the two entities were seen together."""

    subject: str
    object: str
    predicate: str = "co_occurs_with"
    weight: float = 1.0
    sources: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


@runtime_checkable
class EntityExtractor(Protocol):
    """Pulls entities out of text. Swap for a model driven one when available."""

    name: str

    def extract(self, text: str) -> list[Entity]: ...


class PatternEntityExtractor:
    """Deterministic extraction by pattern. No dependencies, no cost, no network."""

    name = "pattern"

    def __init__(self, min_length: int = 3) -> None:
        self.min_length = min_length

    def extract(self, text: str) -> list[Entity]:
        found: dict[str, Entity] = {}

        for match in _IDENTIFIER.finditer(text):
            value = match.group(0).strip()
            if len(value) >= self.min_length and value.lower() not in _NOT_ENTITIES:
                found.setdefault(value.lower(), Entity(name=value, kind="identifier"))

        for match in _PROPER_NOUN.finditer(text):
            value = _strip_article(match.group(1).strip())
            if len(value) < self.min_length:
                continue
            if value.lower() in _NOT_ENTITIES:
                continue
            if all(word.lower() in _NOT_ENTITIES for word in value.split()):
                continue
            # Every sentence starts with a capital, so a single capitalised word
            # in first position carries no evidence of being a name. "Only",
            # "Refunds" and "Staff" all arrive this way and would otherwise
            # become nodes, connecting unrelated documents through grammar rather
            # than through meaning. An all caps token in first position is kept,
            # since that capitalisation is not explained by position.
            if match.start() == 0 and " " not in value and not value.isupper():
                continue
            found.setdefault(value.lower(), Entity(name=value, kind="proper_noun"))

        return list(found.values())


@dataclass
class GraphNeighbourhood:
    """What a traversal found, ready to be turned into prompt context."""

    seeds: list[str]
    entities: list[str]
    relations: list[Relation]
    sources: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.relations


class KnowledgeGraph:
    """An in memory property graph over the corpus.

    Held in memory and rebuilt from the vector store like the keyword index, for
    the same reason: the graph must never be silently out of date relative to the
    documents it claims to describe.
    """

    def __init__(self, extractor: EntityExtractor | None = None) -> None:
        self.extractor: EntityExtractor = extractor or PatternEntityExtractor()
        self._entities: dict[str, Entity] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._entity_sources: dict[str, set[str]] = defaultdict(set)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()
        self._adjacency.clear()
        self._entity_sources.clear()

    def add_document(self, text: str, source: str) -> int:
        """Index one document. Returns the number of relations added or reinforced.

        Entities are linked when they appear in the same sentence rather than the
        same document. Document level co-occurrence connects everything to
        everything on any reasonably sized page, which produces a dense graph that
        carries no information.
        """
        added = 0
        for sentence in (part.strip() for part in _SENTENCE.split(text)):
            if len(sentence) < 15:
                continue
            entities = self.extractor.extract(sentence)
            if len(entities) < 2:
                for entity in entities:
                    self._register(entity, source)
                continue
            for entity in entities:
                self._register(entity, source)
            for i, left in enumerate(entities):
                for right in entities[i + 1 :]:
                    added += self._link(left, right, source, sentence)
        return added

    def _register(self, entity: Entity, source: str) -> None:
        self._entities.setdefault(entity.key, entity)
        self._entity_sources[entity.key].add(source)

    def _link(self, left: Entity, right: Entity, source: str, evidence: str) -> int:
        # Ordered so the pair maps to one edge regardless of sentence order.
        first, second = sorted([left.key, right.key])
        key = (first, "co_occurs_with", second)
        relation = self._relations.get(key)
        if relation is None:
            relation = Relation(subject=first, object=second)
            self._relations[key] = relation
            self._adjacency[first].add(second)
            self._adjacency[second].add(first)
            new = 1
        else:
            relation.weight += 1.0
            new = 0
        relation.sources.add(source)
        if len(relation.evidence) < 3 and evidence not in relation.evidence:
            relation.evidence.append(evidence)
        return new

    def entity(self, key: str) -> Entity | None:
        return self._entities.get(key.lower())

    def neighbours(self, key: str) -> set[str]:
        return set(self._adjacency.get(key.lower(), set()))

    def seeds_for(self, query: str) -> list[str]:
        """Entities in the query that the graph knows about."""
        mentioned = {entity.key for entity in self.extractor.extract(query)}
        # Also match known entities by substring, so "refund" finds "Refund Policy"
        # even though a lowercase query term is not itself extracted as an entity.
        lowered = query.lower()
        for key in self._entities:
            if key in lowered:
                mentioned.add(key)
        return sorted(key for key in mentioned if key in self._entities)

    def traverse(self, query: str, hops: int = 2, max_entities: int = 25) -> GraphNeighbourhood:
        """Walk outward from the entities named in the query.

        Two hops is the default because one hop returns only what a single
        sentence already said, which retrieval covers, while three or more pulls
        in most of a small graph and stops being selective.
        """
        seeds = self.seeds_for(query)
        if not seeds:
            return GraphNeighbourhood(seeds=[], entities=[], relations=[], sources=[])

        visited: set[str] = set(seeds)
        frontier: set[str] = set(seeds)
        for _ in range(max(0, hops)):
            next_frontier: set[str] = set()
            for key in frontier:
                for neighbour in self._adjacency.get(key, set()):
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
            if len(visited) >= max_entities:
                break

        collected = sorted(visited)[:max_entities]
        collected_set = set(collected)
        relations = [
            relation
            for relation in self._relations.values()
            if relation.subject in collected_set and relation.object in collected_set
        ]
        relations.sort(key=lambda relation: (-relation.weight, relation.subject))

        sources: set[str] = set()
        for relation in relations:
            sources.update(relation.sources)

        return GraphNeighbourhood(
            seeds=seeds,
            entities=[self._entities[key].name for key in collected if key in self._entities],
            relations=relations,
            sources=sorted(sources),
        )

    def describe(self) -> dict[str, object]:
        return {
            "extractor": self.extractor.name,
            "entities": self.entity_count,
            "relations": self.relation_count,
        }


def format_graph_context(neighbourhood: GraphNeighbourhood, max_relations: int = 12) -> str:
    """Render a neighbourhood as prompt context.

    Relations are rendered with the sentence that produced them rather than as
    bare triples, because a triple alone gives a model nothing to quote and
    nothing for a reader to check.
    """
    if neighbourhood.is_empty:
        return ""
    lines = [f"Related to: {', '.join(neighbourhood.seeds)}"]
    for relation in neighbourhood.relations[:max_relations]:
        evidence = relation.evidence[0] if relation.evidence else ""
        source = sorted(relation.sources)[0] if relation.sources else "unknown"
        lines.append(f"- {relation.subject} and {relation.object} [{source}]: {evidence}")
    return "\n".join(lines)
