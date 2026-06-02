"""Qdrant client wrapper: connection, collection setup, and CRUD helpers.

Collections are organised along two axes:
- **Domain** — `shared`, `dev`, or `personal`. Endurance writes per-domain;
  OpenClaw and existing callers default to `shared`.
- **Kind** — `entities`, `events`, or `decisions`. The original storage shape.

`shared` is the legacy default and uses the original collection names
(`entities`, `events`, `decisions`) for backward compatibility. Other
domains use `<domain>_<kind>` (e.g. `dev_entities`, `personal_events`).

All public CRUD helpers accept an optional `domain="shared"` kwarg, so
existing call sites continue to work unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from entity_memory.models import Entity, Event, Fact

DOMAINS = ["shared", "dev", "personal"]
COLLECTION_KINDS = ["entities", "events", "decisions"]
VECTOR_SIZE = 384


def collection_name(domain: str, kind: str) -> str:
    """Resolve a (domain, kind) pair to a Qdrant collection name.

    `shared` keeps the legacy bare names so OpenClaw data isn't moved.
    Other domains are namespaced with `<domain>_<kind>`.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")
    if kind not in COLLECTION_KINDS:
        raise ValueError(f"Unknown kind {kind!r}; expected one of {COLLECTION_KINDS}")
    if domain == "shared":
        return kind
    return f"{domain}_{kind}"


ALL_COLLECTIONS = [collection_name(d, k) for d in DOMAINS for k in COLLECTION_KINDS]

# Back-compat alias for callers that imported the legacy 3-collection list.
COLLECTIONS = [collection_name("shared", k) for k in COLLECTION_KINDS]


def load_config() -> dict:
    """Load config from ~/.openclaw/memory.json, falling back to defaults."""
    config_path = Path.home() / ".openclaw" / "memory.json"
    defaults = {
        "qdrant": {"url": "http://127.0.0.1:6333", "api_key_env": "QDRANT_API_KEY"},
    }
    if config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
        defaults.update(user_config)
    return defaults


def get_client(config: dict | None = None) -> QdrantClient:
    """Create a QdrantClient from config."""
    config = config or load_config()
    qdrant_cfg = config.get("qdrant", {})
    url = qdrant_cfg.get("url", "http://127.0.0.1:6333")
    api_key_env = qdrant_cfg.get("api_key_env", "QDRANT_API_KEY")
    api_key = os.environ.get(api_key_env)
    return QdrantClient(url=url, api_key=api_key)


def ensure_collections(client: QdrantClient) -> list[str]:
    """Create all (domain, kind) collections + indexes if missing.

    Returns the list of collection names that were freshly created.
    """
    created = []
    for name in ALL_COLLECTIONS:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            client.create_payload_index(
                collection_name=name,
                field_name="search_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                ),
            )
            client.create_payload_index(name, "type", field_schema=PayloadSchemaType.KEYWORD)
            client.create_payload_index(name, "entity_id", field_schema=PayloadSchemaType.KEYWORD)
            created.append(name)
    return created


def entity_point_id(entity_id: str) -> str:
    """Deterministic UUID from entity_id for Qdrant point ID."""
    h = hashlib.sha256(entity_id.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def entity_to_point(entity: Entity, vector: list[float]) -> PointStruct:
    """Convert an Entity to a Qdrant PointStruct."""
    from entity_memory.merge import build_search_text

    payload = {
        "entity_id": entity.id,
        "type": entity.type,
        "search_text": build_search_text(entity),
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
        "last_updated": entity.last_updated,
    }
    return PointStruct(
        id=entity_point_id(entity.id),
        vector=vector,
        payload=payload,
    )


def point_to_entity(point) -> Entity:
    """Convert a Qdrant point (with payload) back to an Entity."""
    p = point.payload
    facts = [
        Fact(
            text=f["text"],
            added=f["added"],
            source=f["source"],
            expires=f.get("expires"),
            last_seen=f.get("last_seen"),
            hit_count=f.get("hit_count", 1),
        )
        for f in p.get("facts", [])
    ]
    return Entity(
        id=p["entity_id"],
        type=p["type"],
        facts=facts,
        last_updated=p.get("last_updated", ""),
    )


def get_entity(
    client: QdrantClient, entity_id: str, *, domain: str = "shared",
) -> Optional[Entity]:
    """Retrieve a single entity by ID from the given domain. Returns None if missing.

    Decisions and entities live in different collections; this checks both within
    the same domain.
    """
    point_id = entity_point_id(entity_id)
    primary = collection_name(domain, "decisions" if entity_id.startswith("decision:") else "entities")
    fallback = collection_name(domain, "entities" if primary.endswith("decisions") else "decisions")

    if client.collection_exists(primary):
        points = client.retrieve(collection_name=primary, ids=[point_id], with_payload=True)
        if points:
            return point_to_entity(points[0])
    if client.collection_exists(fallback):
        points = client.retrieve(collection_name=fallback, ids=[point_id], with_payload=True)
        if points:
            return point_to_entity(points[0])
    return None


def upsert_entity(
    client: QdrantClient, entity: Entity, vector: list[float], *, domain: str = "shared",
) -> None:
    """Upsert an entity into the appropriate (domain, kind) collection."""
    kind = "decisions" if entity.type == "decision" else "entities"
    point = entity_to_point(entity, vector)
    client.upsert(collection_name=collection_name(domain, kind), points=[point])


def delete_entity(
    client: QdrantClient, entity_id: str, *, domain: str = "shared",
) -> bool:
    """Delete an entity by ID from the given domain. Returns True if found and deleted."""
    point_id = entity_point_id(entity_id)
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        points = client.retrieve(collection_name=coll, ids=[point_id], with_payload=False)
        if points:
            client.delete(collection_name=coll, points_selector=[point_id])
            return True
    return False


def scroll_entities(
    client: QdrantClient, entity_type: str | None = None, *, domain: str = "shared",
) -> list[Entity]:
    """Scroll through entities + decisions in the given domain, optionally filtered by type."""
    results: list[Entity] = []
    for kind in ("entities", "decisions"):
        coll = collection_name(domain, kind)
        if not client.collection_exists(coll):
            continue
        scroll_filter = None
        if entity_type:
            scroll_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=entity_type))]
            )
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=coll,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                results.append(point_to_entity(p))
            if next_offset is None:
                break
            offset = next_offset
    return results


def store_event(
    client: QdrantClient, event_id: str, text: str, vector: list[float],
    timestamp: str, source: str = "conversation", agent: str = "main",
    expires: str | None = None,
    *,
    domain: str = "shared",
    run_id: str | None = None,
    session_id: str | None = None,
    profile: str | None = None,
    trigger_source: str | None = None,
) -> None:
    """Store a raw event in the events collection of the given domain.

    Provenance fields (run_id/session_id/profile/trigger_source) are
    optional; legacy CLI callers leave them None and the payload omits
    them on read for older rows.
    """
    payload = {
        "text": text,
        "timestamp": timestamp,
        "source": source,
        "agent": agent,
        "extracted": False,
        "expires": expires,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    if session_id is not None:
        payload["session_id"] = session_id
    if profile is not None:
        payload["profile"] = profile
    if trigger_source is not None:
        payload["trigger_source"] = trigger_source

    point = PointStruct(id=event_id, vector=vector, payload=payload)
    client.upsert(collection_name=collection_name(domain, "events"), points=[point])


def get_unextracted_events(
    client: QdrantClient, since: datetime | None = None, *, domain: str = "shared",
) -> list[dict]:
    """Get events where extracted=false, optionally filtered by time, in a domain."""
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        return []
    conditions = [FieldCondition(key="extracted", match=MatchValue(value=False))]
    scroll_filter = Filter(must=conditions)
    results: list[dict] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=coll,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for p in points:
            payload = p.payload
            if since is not None:
                ts = datetime.fromisoformat(payload["timestamp"])
                if ts < since:
                    continue
            results.append({"id": p.id, **payload})
        if next_offset is None:
            break
        offset = next_offset
    return results


def mark_event_extracted(
    client: QdrantClient, event_id: str, *, domain: str = "shared",
    resolved_into: str | None = None,
) -> None:
    """Mark an event as extracted within its domain.

    If ``resolved_into`` is given, also records which entity the event fed
    (provenance) on the event payload, so a resolved event points at the
    entity created/updated from it.
    """
    payload: dict = {"extracted": True}
    if resolved_into is not None:
        payload["resolved_into"] = resolved_into
    client.set_payload(
        collection_name=collection_name(domain, "events"),
        payload=payload,
        points=[event_id],
    )


def collection_stats(client: QdrantClient) -> dict[str, dict]:
    """Get point counts and metadata for every (domain, kind) collection."""
    stats: dict[str, dict] = {}
    for name in ALL_COLLECTIONS:
        if client.collection_exists(name):
            info = client.get_collection(name)
            stats[name] = {"points": info.points_count}
        else:
            stats[name] = {"points": 0, "exists": False}
    return stats
