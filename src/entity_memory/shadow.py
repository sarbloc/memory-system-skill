"""Durable shadow work attached to raw events. One consumer, no Jev calls.

Extraction only updates its own fields; it never dequeues shadow work.
Expiry/deletion still applies. This is not a second memory store or a gate.
"""

import uuid
from datetime import datetime, timezone

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from entity_memory.client import collection_name


def shadow_events(client, *, domain: str, since_ms: int, limit: int = 51) -> dict:
    if isinstance(since_ms, bool) or since_ms < 0 or not 1 <= limit <= 51:
        raise ValueError("invalid shadow window or limit")
    coll = collection_name(domain, "events")
    counts = {state: 0 for state in ("pending", "attempted", "done")}
    if not client.collection_exists(coll):
        return {"events": [], "returned": 0, "written": 0, **counts}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    base = [
        FieldCondition(key="shadow_version", match=MatchValue(value=2)),
        FieldCondition(key="shadow_written_at", range=Range(gte=since_ms, lte=now_ms)),
    ]
    for state in counts:
        counts[state] = client.count(
            collection_name=coll, exact=True,
            count_filter=Filter(must=[*base, FieldCondition(
                key="shadow_state", match=MatchValue(value=state),
            )]),
        ).count
    points, _ = client.scroll(
        collection_name=coll, limit=limit, with_vectors=False,
        scroll_filter=Filter(must=[*base, FieldCondition(
            key="shadow_state", match=MatchAny(any=["pending", "attempted"]),
        )]),
        with_payload=["text", "shadow_state", "shadow_written_at", "expires"],
    )
    events = []
    for point in points:
        p = point.payload
        expires = p.get("expires")
        expiry = datetime.fromisoformat(expires) if expires else None
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        # Bound the MCP reply without truncating the meaning of an event.
        # Match the Node consumer's UTF-16 character-unit limit, including
        # supplementary Unicode (Python len counts code points instead).
        oversized = len(p["text"].encode("utf-16-le", errors="surrogatepass")) // 2 > 4000
        events.append({
            "id": str(point.id), "text": None if oversized else p["text"],
            "state": p["shadow_state"], "written_at": p["shadow_written_at"],
            "expires_at": int(expiry.timestamp() * 1000) if expiry is not None else None,
            "expired": expiry is not None and expiry.timestamp() * 1000 <= now_ms,
        })
    return {"events": events, "returned": len(events), "written": sum(counts.values()), **counts}


def shadow_mark(client, *, domain: str, event_id: str, state: str) -> dict:
    """Single-consumer transition; NOT an atomic multi-worker claim.

    attempted is irreversible. An uncertain/crashed attempt is retired, not
    resubmitted. Text/vector/extracted/TTL are untouched. No point is created.
    """
    uuid.UUID(event_id)
    if state not in ("attempted", "done"):
        raise ValueError("invalid shadow state")
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        return {"id": event_id, "state": "missing", "updated": False}
    points = client.retrieve(coll, ids=[event_id], with_vectors=False,
                             with_payload=["shadow_version", "shadow_state"])
    if not points or points[0].payload.get("shadow_version") != 2:
        return {"id": event_id, "state": "missing", "updated": False}
    current = points[0].payload.get("shadow_state")
    if current not in ("pending", "attempted", "done"):
        raise ValueError("invalid stored shadow state")
    if state == "attempted" and current != "pending":
        return {"id": event_id, "state": current, "updated": False}
    client.set_payload(coll, payload={"shadow_state": state}, points=[event_id], wait=True)
    return {"id": event_id, "state": state, "updated": True}
