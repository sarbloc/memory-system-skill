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


def _project_temporal(entity: Entity, as_of: str | None) -> Entity:
    """Return ``entity`` with its facts projected to a temporal view (issue #21).

    Default (``as_of is None``): drop superseded facts so recall reflects current
    state. With ``as_of`` set: keep only facts in effect on that ISO date, so the
    caller sees the entity as it was then. Mutates the (throwaway) entity built
    by ``point_to_entity`` — safe because each search rebuilds entities freshly.

    Note: ranking still comes from the stored "now" vector; only the *returned
    facts* are time-shifted. Reconstructing historical ranking would need a
    historical vector, which we deliberately don't keep.
    """
    if as_of is None:
        entity.facts = [f for f in entity.facts if f.is_current]
    else:
        entity.facts = [f for f in entity.facts if f.valid_at(as_of)]
    return entity


def _collection_size(client: QdrantClient, coll: str) -> int:
    """Exact point count for a collection, or 0 if it doesn't exist/errors."""
    try:
        return client.count(collection_name=coll, exact=True).count
    except Exception:
        return 0


def search_entities(
    client: QdrantClient,
    query: str,
    embedder: EmbedderLike,
    entity_type: str | None = None,
    limit: int = 5,
    *,
    domain: str = "shared",
    as_of: str | None = None,
) -> list[SearchResult]:
    """Search across entities and decisions collections within a domain.

    Uses dense vector search as primary, with optional type filter.
    Deduplicates results across collections by entity_id, keeping the higher score.

    Each result's facts are projected to a temporal view: superseded facts are
    hidden by default; pass ``as_of`` (ISO date) to see each entity as it was on
    that date (issue #21). Entities with no facts valid at that date are dropped,
    and the limit is applied AFTER projection so they don't squat the top slots.
    Use ``memory_get`` for the full, unfiltered record.
    """
    query_vector = embedder.embed(query)

    search_filter = None
    if entity_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
        )

    # Temporal projection can empty an entity's fact list, so we must project and
    # drop the empties BEFORE applying the caller's limit (issue #21). For a
    # historical ``as_of`` query the valid set may be a small slice of a large
    # history, so we rank the whole (small) collection to avoid under-returning
    # when many top-ranked hits are empty then; the default "now" view rarely
    # empties, so a modest over-fetch suffices (Codex review of PR #23).
    if as_of is not None:
        fetch_limit = max(
            (_collection_size(client, collection_name(domain, k))
             for k in ("entities", "decisions")),
            default=limit,
        )
        fetch_limit = max(fetch_limit, limit)
    else:
        fetch_limit = max(limit * 2, limit)

    results_map: dict[str, SearchResult] = {}

    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue

        hits = client.query_points(
            collection_name=coll,
            query=query_vector,
            query_filter=search_filter,
            limit=fetch_limit,
            with_payload=True,
        )

        for hit in hits.points:
            entity = point_to_entity(hit)
            eid = entity.id
            if eid not in results_map or hit.score > results_map[eid].score:
                results_map[eid] = SearchResult(entity=entity, score=hit.score)

    text_results = _text_search(client, query, search_filter, fetch_limit, domain=domain)
    for sr in text_results:
        eid = sr.entity.id
        if eid not in results_map:
            results_map[eid] = sr

    ranked = sorted(results_map.values(), key=lambda r: r.score, reverse=True)

    projected: list[SearchResult] = []
    for r in ranked:
        _project_temporal(r.entity, as_of)
        if r.entity.facts:
            projected.append(r)
        if len(projected) >= limit:
            break
    return projected


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
