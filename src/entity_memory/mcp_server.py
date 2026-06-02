"""MCP stdio server exposing entity-memory operations as tools.

Started by `memory mcp`. Talks the Model Context Protocol over stdio so an
MCP client (e.g. Endurance, Claude Code, etc.) can invoke memory operations
without shelling out to the CLI.

Tools mirror the existing CLI verbs and add an explicit `domain` parameter
for routing between the per-domain Qdrant collections defined in client.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

from entity_memory.client import (
    DOMAINS,
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    get_unextracted_events,
    mark_event_extracted,
    scroll_entities,
    store_event,
    upsert_entity,
)
from entity_memory.extract import extract_events
from entity_memory.merge import build_search_text, compact, merge
from entity_memory.models import Entity, Fact
from entity_memory.search import format_results, search_entities

EVENT_TTL_DAYS = 30

mcp = FastMCP("entity-memory")

_embedder = None


def _embedder_instance():
    """Lazy-load the embedder so import time stays cheap."""
    global _embedder
    if _embedder is None:
        from entity_memory.embedder import Embedder

        _embedder = Embedder()
    return _embedder


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.type,
        "last_updated": entity.last_updated,
        "facts": [
            {
                "text": f.text,
                "added": f.added,
                "source": f.source,
                "expires": f.expires,
                "last_seen": f.last_seen,
                "hit_count": f.hit_count,
            }
            for f in entity.facts
        ],
    }


def _validate_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")


@mcp.tool()
def memory_store(
    entity_type: str,
    entity_id: str,
    content: str,
    domain: str = "shared",
) -> dict[str, Any]:
    """Upsert an entity with a fact in the given domain.

    Args:
        entity_type: person | project | tool | preference | decision
        entity_id: short id (e.g. 'alice') or full id ('person:alice')
        content: the fact text
        domain: shared | dev | personal (default shared)
    """
    _validate_domain(domain)
    client = get_client()
    ensure_collections(client)
    embedder = _embedder_instance()

    full_id = f"{entity_type}:{entity_id}" if ":" not in entity_id else entity_id
    now = datetime.utcnow()
    today = now.date().isoformat()

    existing = get_entity(client, full_id, domain=domain)
    if existing is None:
        existing = Entity(id=full_id, type=entity_type)

    new_fact = Fact(text=content, added=today, source="mcp:store")
    merged = merge(existing, [new_fact], embedder, now=now)
    vector = embedder.embed(build_search_text(merged))
    upsert_entity(client, merged, vector, domain=domain)

    return {"id": full_id, "facts": len(merged.facts), "domain": domain}


@mcp.tool()
def memory_get(entity_id: str, domain: str = "shared") -> dict[str, Any] | None:
    """Retrieve a single entity by id from a domain. Returns None if missing."""
    _validate_domain(domain)
    client = get_client()
    entity = get_entity(client, entity_id, domain=domain)
    return _entity_to_dict(entity) if entity else None


@mcp.tool()
def memory_list(
    domain: str = "shared",
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """List entities in a domain, optionally filtered by type."""
    _validate_domain(domain)
    client = get_client()
    entities = scroll_entities(client, entity_type=entity_type, domain=domain)
    return [
        {"id": e.id, "type": e.type, "facts": len(e.facts), "last_updated": e.last_updated}
        for e in entities
    ]


@mcp.tool()
def memory_search(
    query: str,
    domain: str = "shared",
    entity_type: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Semantic search across entities and decisions in a domain.

    Returns hits sorted by score, each with the entity payload and score.
    """
    _validate_domain(domain)
    client = get_client()
    embedder = _embedder_instance()
    results = search_entities(
        client, query, embedder, entity_type=entity_type, limit=limit, domain=domain,
    )
    return [
        {"score": r.score, **_entity_to_dict(r.entity)}
        for r in results
    ]


@mcp.tool()
def memory_event(
    text: str,
    domain: str = "shared",
    source: str = "conversation",
    agent: str = "main",
    run_id: str | None = None,
    session_id: str | None = None,
    profile: str | None = None,
    trigger_source: str | None = None,
) -> dict[str, Any]:
    """Log a raw observation. No entity extraction happens immediately.

    Provenance fields (run_id/session_id/profile/trigger_source) are
    persisted on the event payload for later filtering.
    """
    _validate_domain(domain)
    client = get_client()
    ensure_collections(client)
    embedder = _embedder_instance()

    now = datetime.utcnow()
    event_id = str(uuid.uuid4())
    vector = embedder.embed(text)
    expires = (now + timedelta(days=EVENT_TTL_DAYS)).isoformat()

    store_event(
        client,
        event_id=event_id,
        text=text,
        vector=vector,
        timestamp=now.isoformat(),
        source=source,
        agent=agent,
        expires=expires,
        domain=domain,
        run_id=run_id,
        session_id=session_id,
        profile=profile,
        trigger_source=trigger_source,
    )
    return {"id": event_id, "domain": domain}


@mcp.tool()
def memory_extract(
    domain: str = "shared",
    since_minutes: int | None = None,
    process_all: bool = False,
) -> dict[str, Any]:
    """Process unextracted events into entity upserts within a domain."""
    _validate_domain(domain)
    client = get_client()
    embedder = _embedder_instance()

    since_dt = None
    if since_minutes is not None and not process_all:
        since_dt = datetime.utcnow() - timedelta(minutes=since_minutes)

    events = get_unextracted_events(client, since=since_dt, domain=domain)
    if not events:
        return {
            "events_processed": 0,
            "matched": 0,
            "unmatched": 0,
            "matched_events": 0,
            "unmatched_samples": [],
        }

    entities = scroll_entities(client, domain=domain)
    now = datetime.utcnow()
    today = now.date().isoformat()
    result = extract_events(events, entities, embedder, now=now)

    for sentence, entity in result.matched:
        new_fact = Fact(text=sentence, added=today, source="extract")
        merged = merge(entity, [new_fact], embedder, now=now)
        vector = embedder.embed(build_search_text(merged))
        upsert_entity(client, merged, vector, domain=domain)

    # Mark extracted ONLY events that enriched an existing entity (>=1 sentence
    # matched). Fully-unmatched events stay unextracted so an external LLM agent
    # can turn them into new entities via memory_store, then call
    # memory_event_resolve. (Tradeoff: an event with a MIX of matched and
    # unmatched sentences is marked extracted because >=1 matched, so its
    # unmatched sentences are not separately surfaced. Acceptable for now.)
    for event_id in result.matched_event_ids:
        mark_event_extracted(client, event_id, domain=domain)

    return {
        "events_processed": result.events_processed,
        "matched": len(result.matched),
        "unmatched": len(result.unmatched),
        "matched_events": len(result.matched_event_ids),
        "unmatched_samples": result.unmatched[:10],
    }


@mcp.tool()
def memory_events_unextracted(domain: str = "shared", limit: int = 50) -> dict[str, Any]:
    """List raw events not yet extracted into entities, oldest first, within a domain.

    Use this to find events that need turning into entities (via memory_store).
    Returns {"events": [{"id": str, "text": str, "timestamp": str}], "returned": int}.
    """
    _validate_domain(domain)
    client = get_client()
    events = get_unextracted_events(client, since=None, domain=domain)
    events.sort(key=lambda e: e.get("timestamp", ""))
    sliced = events[:limit]
    projected = [
        {"id": e["id"], "text": e["text"], "timestamp": e.get("timestamp", "")}
        for e in sliced
    ]
    return {"events": projected, "returned": len(projected)}


@mcp.tool()
def memory_event_resolve(
    event_id: str, domain: str = "shared", created_entity_id: str | None = None,
) -> dict[str, Any]:
    """Mark a raw event as extracted/resolved, optionally recording which entity it fed.

    Call after creating/updating an entity from the event (pass created_entity_id),
    or to dismiss a non-memory-worthy event (omit created_entity_id).
    Returns {"resolved": event_id, "created_entity_id": created_entity_id}.
    """
    _validate_domain(domain)
    client = get_client()
    mark_event_extracted(
        client, event_id, domain=domain, resolved_into=created_entity_id,
    )
    return {"resolved": event_id, "created_entity_id": created_entity_id}


@mcp.tool()
def memory_embed(text: str) -> list[float]:
    """Return the MiniLM embedding (384 dims) for a string.

    Exposed so consumers (e.g. Endurance's per-turn auto-injection) can
    use the same embedder rather than spinning up their own model.
    """
    return _embedder_instance().embed(text)


@mcp.tool()
def memory_stats() -> dict[str, dict[str, Any]]:
    """Per-collection point counts for all (domain, kind) collections."""
    client = get_client()
    return collection_stats(client)


def run() -> None:
    """Entry point used by `memory mcp`."""
    mcp.run()


if __name__ == "__main__":
    run()
