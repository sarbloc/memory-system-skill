"""Tests for the `memory extract` CLI command.

Focus is the burn-fix (issue #14): the CLI path must mark extracted ONLY the
events that enriched an existing entity, leaving fully-unmatched events as
new-entity candidates — mirroring the #13 fix already covered for the MCP path
in test_mcp_server.py. Also covers the new --domain routing.

Uses an in-memory Qdrant + MockEmbedder wired into the cli module (same spirit
as test_mcp_server's mcp_env fixture), driven through Click's CliRunner so the
actual command orchestration is exercised end to end.
"""

import uuid

import pytest
from click.testing import CliRunner

import entity_memory.cli as cli_mod
from entity_memory.cli import main
from entity_memory.client import (
    ALL_COLLECTIONS,
    get_unextracted_events,
    store_event,
    upsert_entity,
)
from entity_memory.extract import cosine_sim
from entity_memory.merge import build_search_text
from entity_memory.models import Entity, Fact

TODAY = "2026-03-10"
NOW_ISO = "2026-03-10T12:00:00"


@pytest.fixture
def cli_env(embedder, monkeypatch):
    """In-memory Qdrant + mock embedder wired into the cli module.

    Collections are created at the mock embedder's dimension (not via
    ensure_collections, which hardcodes the 384-dim real model size).
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

    monkeypatch.setattr(cli_mod, "get_client", lambda *a, **k: client)
    monkeypatch.setattr(cli_mod, "_get_embedder", lambda *a, **k: embedder)
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


def test_unmatched_event_not_burned(cli_env, embedder):
    """The core #14 fix: a fully-unmatched event survives an extract run."""
    client = cli_env
    _, st = _seed_entity(
        client, embedder, "person:alice", "person", ["Manages the auth team"]
    )

    # A sentence equal to the entity's search_text matches deterministically.
    matched_id = _seed_event(client, embedder, st)

    # A fully-unmatched event: cosine < threshold vs the only entity present.
    unmatched_text = "grok quaffle."
    if cosine_sim(embedder.embed(unmatched_text), embedder.embed(st)) >= 0.7:
        pytest.skip("Mock embedder changed: chosen text now matches")
    unmatched_id = _seed_event(client, embedder, unmatched_text)

    result = CliRunner().invoke(main, ["extract", "--all"])
    assert result.exit_code == 0, result.output
    assert "Processed 2 events" in result.output

    remaining = {e["id"] for e in get_unextracted_events(client)}
    assert unmatched_id in remaining  # NOT burned — still a candidate
    assert matched_id not in remaining  # matched → correctly extracted


def test_domain_flag_routes_to_domain(cli_env, embedder):
    """--domain selects the collection set; default (shared) leaves dev alone."""
    client = cli_env
    dev_id = _seed_event(client, embedder, "grok quaffle.", domain="dev")

    # Default-shared extract sees nothing and must not touch the dev event.
    shared_run = CliRunner().invoke(main, ["extract", "--all"])
    assert shared_run.exit_code == 0, shared_run.output
    assert "No unextracted events found." in shared_run.output
    assert dev_id in {e["id"] for e in get_unextracted_events(client, domain="dev")}

    # --domain dev operates on the dev collection and sees the event.
    dev_run = CliRunner().invoke(main, ["extract", "--all", "--domain", "dev"])
    assert dev_run.exit_code == 0, dev_run.output
    assert "Processed 1 events" in dev_run.output


def test_invalid_domain_rejected(cli_env):
    """Click rejects an out-of-set domain before any work happens."""
    result = CliRunner().invoke(main, ["extract", "--all", "--domain", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_batch_flag_reports_progress(cli_env, embedder):
    """--batch chunks the run and emits a per-batch progress line."""
    client = cli_env
    _, st = _seed_entity(client, embedder, "person:alice", "person", ["Manages auth"])
    for _ in range(3):
        _seed_event(client, embedder, st)  # all match the entity

    result = CliRunner().invoke(main, ["extract", "--all", "--batch", "1"])
    assert result.exit_code == 0, result.output
    assert "Processed 3 events" in result.output
    assert "batch 1/3" in result.output  # progress emitted per batch
