"""Semantic search across entity collections with vector + text + filter fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    SearchParams,
)

from entity_memory.client import collection_name, point_to_entity
from entity_memory.models import Entity


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class SearchResult:
    entity: Entity
    score: float


def search_entities(
    client: QdrantClient,
    query: str,
    embedder: EmbedderLike,
    entity_type: str | None = None,
    limit: int = 5,
    *,
    domain: str = "shared",
) -> list[SearchResult]:
    """Search across entities and decisions collections within a domain.

    Uses dense vector search as primary, with optional type filter.
    Deduplicates results across collections by entity_id, keeping the higher score.
    """
    query_vector = embedder.embed(query)

    search_filter = None
    if entity_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
        )

    results_map: dict[str, SearchResult] = {}

    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue

        hits = client.query_points(
            collection_name=coll,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        for hit in hits.points:
            entity = point_to_entity(hit)
            eid = entity.id
            if eid not in results_map or hit.score > results_map[eid].score:
                results_map[eid] = SearchResult(entity=entity, score=hit.score)

    text_results = _text_search(client, query, search_filter, limit, domain=domain)
    for sr in text_results:
        eid = sr.entity.id
        if eid not in results_map:
            results_map[eid] = sr

    results = sorted(results_map.values(), key=lambda r: r.score, reverse=True)
    return results[:limit]


def _text_search(
    client: QdrantClient,
    query: str,
    extra_filter: Filter | None,
    limit: int,
    *,
    domain: str = "shared",
) -> list[SearchResult]:
    """Keyword fallback: search the text index on search_text field within a domain."""
    results = []
    text_condition = FieldCondition(key="search_text", match=MatchText(text=query))

    conditions = [text_condition]
    if extra_filter and extra_filter.must:
        conditions.extend(extra_filter.must)

    scroll_filter = Filter(must=conditions)

    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        points, _ = client.scroll(
            collection_name=coll,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        for p in points:
            entity = point_to_entity(p)
            results.append(SearchResult(entity=entity, score=0.5))

    return results


def format_results(results: list[SearchResult]) -> str:
    """Format search results for CLI output."""
    if not results:
        return "No results found."

    lines = []
    for r in results:
        # Build a summary from top facts
        facts_summary = ". ".join(f.text for f in r.entity.facts[:3])
        lines.append(f"[{r.score:.2f}] {r.entity.id} — {facts_summary}")
    return "\n".join(lines)
