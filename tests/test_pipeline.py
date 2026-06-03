"""Tests for the shared extraction pipeline (run_extraction).

Covers the issue #12 behaviours that the CLI and MCP paths now share:
  - batching (events split into batch_size chunks),
  - per-batch commits (a kill mid-run preserves completed batches),
  - progress callback fired once per batch,
  - the #14 burn-fix holding through batching (unmatched events survive).

Uses an in-memory Qdrant + MockEmbedder wired through the real client helpers,
same fixture style as test_mcp_server.py.
"""

import uuid

import pytest

from entity_memory.client import (
    ALL_COLLECTIONS,
    get_unextracted_events,
    store_event,
    upsert_entity,
)
from entity_memory.extract import cosine_sim
from entity_memory.merge import build_search_text
from entity_memory.models import Entity, Fact
from entity_memory.pipeline import run_extraction

TODAY = "2026-03-10"
NOW_ISO = "2026-03-10T12:00:00"


@pytest.fixture
def store(embedder):
    """In-memory Qdrant with all collections created at the mock dimension."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    dim = len(embedder.embed("dimension probe"))
    client = QdrantClient(":memory:")
    for name in ALL_COLLECTIONS:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    return client


def _seed_event(client, embedder, text, *, domain="shared", timestamp=NOW_ISO):
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
    facts = [Fact(text=t, added=TODAY, source="test", last_seen=TODAY) for t in fact_texts]
    entity = Entity(id=eid, type=etype, facts=facts, last_updated=NOW_ISO)
    st = build_search_text(entity)
    upsert_entity(client, entity, embedder.embed(st), domain=domain)
    return entity, st


def test_empty_events_returns_zero_summary(store, embedder):
    assert run_extraction(store, embedder) == {
        "events_processed": 0,
        "matched": 0,
        "unmatched": 0,
        "matched_events": 0,
        "unmatched_samples": [],
    }


def test_batch_size_must_be_positive(store, embedder):
    with pytest.raises(ValueError):
        run_extraction(store, embedder, batch_size=0)


def test_burn_fix_holds_with_batching(store, embedder):
    """A fully-unmatched event survives even when batched one-per-batch."""
    # "Manages the auth team" is the entity text proven not to collide with
    # "grok quaffle." under the hash-based MockEmbedder (see test_extract.py).
    _, st = _seed_entity(store, embedder, "person:alice", "person", ["Manages the auth team"])
    matched_id = _seed_event(store, embedder, st)  # == search_text → matches

    unmatched_text = "grok quaffle."
    if cosine_sim(embedder.embed(unmatched_text), embedder.embed(st)) >= 0.7:
        pytest.skip("Mock embedder changed: chosen text now matches")
    unmatched_id = _seed_event(store, embedder, unmatched_text)

    summary = run_extraction(store, embedder, batch_size=1)

    assert summary["events_processed"] == 2
    assert summary["matched_events"] == 1
    remaining = {e["id"] for e in get_unextracted_events(store)}
    assert unmatched_id in remaining  # not burned
    assert matched_id not in remaining  # extracted


def test_progress_called_once_per_batch(store, embedder):
    _, st = _seed_entity(store, embedder, "person:alice", "person", ["Manages auth"])
    for _ in range(5):
        _seed_event(store, embedder, st)

    seen = []
    run_extraction(store, embedder, batch_size=2, progress=seen.append)

    # 5 events, batch_size 2 → 3 batches (2, 2, 1).
    assert len(seen) == 3
    assert [info["batch"] for info in seen] == [1, 2, 3]
    assert seen[0]["total_batches"] == 3
    assert seen[-1]["cumulative_events"] == 5
    assert seen[-1]["total_events"] == 5


def test_per_batch_commit_survives_midrun_failure(store, embedder):
    """If the run dies mid-way, batches that already completed stay committed.

    With batch_size=1 and a progress callback that raises on the 2nd batch,
    batches 1 and 2 mark their matched event extracted *before* progress fires,
    so exactly two events are committed; the remaining batches never run and
    their events survive as candidates.
    """
    _, st = _seed_entity(store, embedder, "person:alice", "person", ["Manages auth"])
    for _ in range(4):
        _seed_event(store, embedder, st)  # all match the entity

    calls = {"n": 0}

    def boom(_info):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-run")

    with pytest.raises(RuntimeError):
        run_extraction(store, embedder, batch_size=1, progress=boom)

    # Two batches committed (marked extracted) before the crash; two survive.
    remaining = get_unextracted_events(store)
    assert len(remaining) == 2


def test_unmatched_samples_capped_at_ten(store, embedder):
    _, st = _seed_entity(store, embedder, "person:alice", "person", ["Manages the auth team"])
    # 15 events that do not match the entity → 15 unmatched sentences.
    unmatched_text = "grok quaffle."
    if cosine_sim(embedder.embed(unmatched_text), embedder.embed(st)) >= 0.7:
        pytest.skip("Mock embedder changed: chosen text now matches")
    for _ in range(15):
        _seed_event(store, embedder, unmatched_text)

    summary = run_extraction(store, embedder, batch_size=4)

    assert summary["unmatched"] == 15
    assert summary["matched_events"] == 0
    assert len(summary["unmatched_samples"]) == 10  # capped


def test_domain_isolation(store, embedder):
    """run_extraction on one domain does not touch another."""
    dev_id = _seed_event(store, embedder, "grok quaffle.", domain="dev")

    summary = run_extraction(store, embedder, domain="shared")
    assert summary["events_processed"] == 0
    # dev event untouched by the shared run.
    assert dev_id in {e["id"] for e in get_unextracted_events(store, domain="dev")}
