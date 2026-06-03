"""Tests for the MCP server tools.

Covers the event→entity pipeline fix:
- memory_extract marks ONLY events that enriched an existing entity, leaving
  fully-unmatched events as new-entity candidates (no longer burned).
- memory_events_unextracted lists those candidates (oldest first, limit).
- memory_event_resolve marks an event extracted + records provenance.

Uses an in-memory Qdrant client and the deterministic MockEmbedder, reusing
the same fixture style as the existing suite (conftest `embedder`). The
mcp_server tools call module-level get_client()/_embedder_instance(); we
monkeypatch both to point at the in-memory store.

Note: local (in-memory) Qdrant requires point ids to be valid UUIDs, so
events are seeded with uuid4 ids — matching what the real memory_event tool
does in production.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from entity_memory import mcp_server
from entity_memory.client import ALL_COLLECTIONS, store_event, upsert_entity
from entity_memory.merge import build_search_text
from entity_memory.models import Entity, Fact

NOW_ISO = "2026-03-10T12:00:00"
TODAY = "2026-03-10"


@pytest.fixture
def mcp_env(embedder, monkeypatch):
    """In-memory Qdrant + mock embedder wired into the mcp_server module.

    Collections are created at the mock embedder's actual vector dimension
    (rather than via ensure_collections, which hardcodes the 384-dim real
    model size). Returns the QdrantClient so tests can assert on the store.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    dim = len(embedder.embed("dimension probe"))
    client = QdrantClient(":memory:")
    for name in ALL_COLLECTIONS:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    monkeypatch.setattr(mcp_server, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(mcp_server, "_embedder_instance", lambda: embedder)
    return client


def _seed_event(client, embedder, text, *, timestamp=NOW_ISO, domain="shared"):
    """Store one event, returning its id (a uuid4 string, as in production)."""
    event_id = str(uuid.uuid4())
    store_event(
        client,
        event_id=event_id,
        text=text,
        vector=embedder.embed(text),
        timestamp=timestamp,
        domain=domain,
    )
    return event_id


def _seed_entity(client, embedder, eid, etype, fact_texts, *, domain="shared"):
    """Upsert an entity; return (entity, its build_search_text)."""
    facts = [Fact(text=t, added=TODAY, source="test", last_seen=TODAY) for t in fact_texts]
    entity = Entity(id=eid, type=etype, facts=facts, last_updated=NOW_ISO)
    st = build_search_text(entity)
    upsert_entity(client, entity, embedder.embed(st), domain=domain)
    return entity, st


# ── memory_extract: stop the burn ────────────────────────

class TestMemoryExtractBurnFix:
    def test_only_matched_events_marked_extracted(self, mcp_env, embedder):
        client = mcp_env
        _, st = _seed_entity(
            client, embedder, "person:alice", "person", ["Manages the auth team"]
        )

        # The blob's fact sentence equals a fact row → matches deterministically.
        matched_id = _seed_event(client, embedder, st)

        # A distinct event that won't match at a 0.95 gate: MockEmbedder cosines
        # floor near 0.7, so only identical text clears 0.95 (see test_pipeline).
        unmatched_id = _seed_event(client, embedder, "grok quaffle.")

        result = mcp_server.memory_extract(
            domain="shared", process_all=True, threshold=0.95
        )

        assert result["events_processed"] == 2
        assert result["matched_events"] == 1

        # The unmatched event must NOT be burned — it still surfaces.
        remaining = {
            e["id"] for e in mcp_server.memory_events_unextracted(domain="shared")["events"]
        }
        assert unmatched_id in remaining
        assert matched_id not in remaining

    def test_no_events_short_circuit(self, mcp_env):
        result = mcp_server.memory_extract(domain="shared", process_all=True)
        assert result == {
            "events_processed": 0,
            "matched": 0,
            "unmatched": 0,
            "matched_events": 0,
            "unmatched_samples": [],
        }


# ── memory_events_unextracted ────────────────────────────

class TestMemoryEventsUnextracted:
    def test_shape(self, mcp_env, embedder):
        client = mcp_env
        eid = _seed_event(client, embedder, "Something happened today.")
        out = mcp_server.memory_events_unextracted(domain="shared")
        assert set(out.keys()) == {"events", "returned"}
        assert out["returned"] == 1
        assert len(out["events"]) == 1
        ev = out["events"][0]
        assert set(ev.keys()) == {"id", "text", "timestamp"}
        assert ev["id"] == eid
        assert ev["text"] == "Something happened today."
        assert ev["timestamp"] == NOW_ISO

    def test_empty(self, mcp_env):
        out = mcp_server.memory_events_unextracted(domain="shared")
        assert out == {"events": [], "returned": 0}

    def test_oldest_first(self, mcp_env, embedder):
        client = mcp_env
        # Seed out of chronological order; expect oldest-first output.
        mid = _seed_event(client, embedder, "middle.", timestamp="2026-02-02T00:00:00")
        old = _seed_event(client, embedder, "oldest.", timestamp="2026-01-01T00:00:00")
        new = _seed_event(client, embedder, "newest.", timestamp="2026-03-03T00:00:00")
        out = mcp_server.memory_events_unextracted(domain="shared")
        ids = [e["id"] for e in out["events"]]
        assert ids == [old, mid, new]

    def test_respects_limit(self, mcp_env, embedder):
        client = mcp_env
        ids = []
        for i in range(5):
            ids.append(
                _seed_event(
                    client, embedder, f"event {i}.",
                    timestamp=f"2026-01-0{i + 1}T00:00:00",
                )
            )
        out = mcp_server.memory_events_unextracted(domain="shared", limit=2)
        assert out["returned"] == 2
        # Limit keeps the two oldest.
        assert [e["id"] for e in out["events"]] == ids[:2]

    def test_since_minutes_filters_old(self, mcp_env, embedder):
        client = mcp_env
        # since_minutes is relative to real utcnow(), so seed timestamps
        # relative to now rather than the fixed NOW_ISO used elsewhere.
        now = datetime.utcnow()
        recent = _seed_event(
            client, embedder, "recent.",
            timestamp=(now - timedelta(minutes=10)).isoformat(),
        )
        old = _seed_event(
            client, embedder, "stale.",
            timestamp=(now - timedelta(days=3)).isoformat(),
        )
        out = mcp_server.memory_events_unextracted(domain="shared", since_minutes=1440)
        ids = [e["id"] for e in out["events"]]
        assert out["returned"] == 1
        assert recent in ids
        assert old not in ids


# ── memory_event_resolve ─────────────────────────────────

class TestMemoryEventResolve:
    def test_marks_extracted(self, mcp_env, embedder):
        client = mcp_env
        eid = _seed_event(client, embedder, "Resolve me.")
        # Present before resolve.
        before = {e["id"] for e in mcp_server.memory_events_unextracted(domain="shared")["events"]}
        assert eid in before

        ret = mcp_server.memory_event_resolve(event_id=eid, domain="shared")
        assert ret == {"resolved": eid, "created_entity_id": None}

        # Gone after resolve.
        after = {e["id"] for e in mcp_server.memory_events_unextracted(domain="shared")["events"]}
        assert eid not in after

    def test_records_provenance(self, mcp_env, embedder):
        client = mcp_env
        eid = _seed_event(client, embedder, "Fed an entity.")
        ret = mcp_server.memory_event_resolve(
            event_id=eid, domain="shared", created_entity_id="person:bob"
        )
        assert ret == {"resolved": eid, "created_entity_id": "person:bob"}

        points = client.retrieve("events", ids=[eid], with_payload=True)
        payload = points[0].payload
        assert payload["extracted"] is True
        assert payload["resolved_into"] == "person:bob"

    def test_invalid_domain_rejected(self, mcp_env, embedder):
        with pytest.raises(ValueError):
            mcp_server.memory_event_resolve(event_id="x", domain="bogus")


# ── bi-temporal supersession via the MCP surface (issue #21) ──

class TestMemoryStoreSupersession:
    def test_supersedes_marks_old_fact(self, mcp_env):
        mcp_server.memory_store("person", "alice", "lives in London", domain="shared")
        out = mcp_server.memory_store(
            "person", "alice", "lives in Berlin",
            domain="shared", supersedes="lives in London",
        )
        assert out["superseded"] == "lives in London"

        # memory_get is the full record — both facts present, old one superseded.
        ent = mcp_server.memory_get("person:alice", domain="shared")
        by_text = {f["text"]: f for f in ent["facts"]}
        assert by_text["lives in London"]["superseded_at"] is not None
        assert by_text["lives in London"]["superseded_by"] == "lives in Berlin"
        assert by_text["lives in Berlin"]["superseded_at"] is None

    def test_supersedes_no_match_reports_none(self, mcp_env):
        out = mcp_server.memory_store(
            "person", "bob", "first fact", domain="shared", supersedes="nonexistent",
        )
        assert out["superseded"] is None

    def test_search_hides_superseded_by_default(self, mcp_env):
        mcp_server.memory_store("person", "alice", "lives in London", domain="shared")
        mcp_server.memory_store(
            "person", "alice", "lives in Berlin",
            domain="shared", supersedes="lives in London",
        )
        hits = mcp_server.memory_search("where does alice live", domain="shared")
        alice = next(h for h in hits if h["id"] == "person:alice")
        texts = {f["text"] for f in alice["facts"]}
        assert "lives in Berlin" in texts
        assert "lives in London" not in texts

    def test_search_as_of_returns_historical_state(self, mcp_env, embedder):
        client = mcp_env
        # Seed with explicit dates so the as-of window is deterministic.
        old = Fact(
            text="lives in London", added="2026-01-01", source="test",
            last_seen="2026-01-01", superseded_at="2026-03-01",
            superseded_by="lives in Berlin",
        )
        new = Fact(
            text="lives in Berlin", added="2026-03-01", source="test",
            last_seen="2026-03-01", valid_from="2026-03-01",
        )
        entity = Entity(id="person:alice", type="person", facts=[old, new], last_updated=NOW_ISO)
        upsert_entity(client, entity, embedder.embed(build_search_text(entity)), domain="shared")

        # As of Feb → only the London fact was in effect.
        hist = mcp_server.memory_search("alice", domain="shared", as_of="2026-02-01")
        alice = next(h for h in hist if h["id"] == "person:alice")
        assert {f["text"] for f in alice["facts"]} == {"lives in London"}

        # Default (now) → only the current Berlin fact.
        cur = mcp_server.memory_search("alice", domain="shared")
        alice_now = next(h for h in cur if h["id"] == "person:alice")
        assert {f["text"] for f in alice_now["facts"]} == {"lives in Berlin"}
