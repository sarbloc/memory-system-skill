"""Qdrant client wrapper: connection, collection setup, and CRUD helpers."""

from __future__ import annotations

import hashlib
import json
import os
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

COLLECTIONS = ["entities", "events", "decisions"]
VECTOR_SIZE = 384


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
    """Create collections and indexes if they don't exist. Returns list of created collections."""
    created = []
    for name in COLLECTIONS:
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
    # Format as UUID: 8-4-4-4-12
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


def get_entity(client: QdrantClient, entity_id: str) -> Optional[Entity]:
    """Retrieve a single entity by ID. Returns None if not found."""
    collection = "decisions" if entity_id.startswith("decision:") or _is_decision(client, entity_id) else "entities"
    points = client.retrieve(
        collection_name=collection,
        ids=[entity_point_id(entity_id)],
        with_payload=True,
    )
    if not points:
        # Try the other collection
        other = "decisions" if collection == "entities" else "entities"
        points = client.retrieve(
            collection_name=other,
            ids=[entity_point_id(entity_id)],
            with_payload=True,
        )
    if not points:
        return None
    return point_to_entity(points[0])


def _is_decision(client: QdrantClient, entity_id: str) -> bool:
    """Check if an entity_id exists in the decisions collection."""
    points = client.retrieve(
        collection_name="decisions",
        ids=[entity_point_id(entity_id)],
        with_payload=False,
    )
    return len(points) > 0


def upsert_entity(client: QdrantClient, entity: Entity, vector: list[float]) -> None:
    """Upsert an entity into the appropriate collection."""
    collection = "decisions" if entity.type == "decision" else "entities"
    point = entity_to_point(entity, vector)
    client.upsert(collection_name=collection, points=[point])


def delete_entity(client: QdrantClient, entity_id: str) -> bool:
    """Delete an entity by ID. Returns True if found and deleted."""
    for collection in ["entities", "decisions"]:
        points = client.retrieve(
            collection_name=collection,
            ids=[entity_point_id(entity_id)],
            with_payload=False,
        )
        if points:
            client.delete(
                collection_name=collection,
                points_selector=[entity_point_id(entity_id)],
            )
            return True
    return False


def collection_stats(client: QdrantClient) -> dict[str, dict]:
    """Get point counts and metadata for all collections."""
    stats = {}
    for name in COLLECTIONS:
        if client.collection_exists(name):
            info = client.get_collection(name)
            stats[name] = {"points": info.points_count}
        else:
            stats[name] = {"points": 0, "exists": False}
    return stats
