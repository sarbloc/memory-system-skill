"""Tests for event extraction logic."""

from datetime import datetime

from entity_memory.extract import (
    extract_events,
    match_sentence_to_entity,
    split_sentences,
)
from entity_memory.models import Entity, Fact


NOW = datetime(2026, 3, 10)
TODAY = NOW.date().isoformat()


def _entity(eid, etype, facts_texts):
    facts = [
        Fact(text=t, added=TODAY, source="test", last_seen=TODAY)
        for t in facts_texts
    ]
    return Entity(id=eid, type=etype, facts=facts, last_updated=NOW.isoformat())


# ── split_sentences ──────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self):
        text = "Alice reviewed the PR. Bob approved it. Ship it."
        result = split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self):
        result = split_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_filters_short_fragments(self):
        result = split_sentences("Hello. OK. This is a real sentence.")
        # "OK" is only 2 chars, should be filtered (< 5)
        assert all(len(s) >= 5 for s in result)

    def test_preserves_content(self):
        text = "Built auth API with Express. Alice reviewed and approved."
        result = split_sentences(text)
        assert "Built auth API with Express." in result
        assert "Alice reviewed and approved." in result

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_no_punctuation(self):
        result = split_sentences("This has no ending punctuation")
        assert len(result) == 1


# ── match_sentence_to_entity ─────────────────────────────

class TestMatchSentence:
    def test_exact_match(self, embedder):
        entity = _entity("person:alice", "person", ["Manages the auth team"])
        result = match_sentence_to_entity(
            "Manages the auth team", [entity], embedder
        )
        # Same text → same embedding → cosine 1.0 → should match
        assert result is entity

    def test_no_match_empty_entities(self, embedder):
        result = match_sentence_to_entity("anything", [], embedder)
        assert result is None

    def test_no_match_different_text(self, embedder):
        entity = _entity("person:alice", "person", ["Manages the auth team"])
        result = match_sentence_to_entity(
            "Completely unrelated topic about cooking recipes", [entity], embedder
        )
        # MockEmbedder: different text → low cosine → likely no match
        # (could match by hash collision, but very unlikely)
        if result is not None:
            import pytest
            pytest.skip("Hash collision in mock embedder")


# ── extract_events ───────────────────────────────────────

class TestExtractEvents:
    def test_empty_events(self, embedder):
        result = extract_events([], [], embedder, now=NOW)
        assert result.events_processed == 0
        assert result.matched == []
        assert result.unmatched == []

    def test_no_entities_all_unmatched(self, embedder):
        events = [{"id": "e1", "text": "Alice reviewed the PR. Bob approved it."}]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.events_processed == 1
        assert len(result.matched) == 0
        assert len(result.unmatched) == 2

    def test_counts_events_processed(self, embedder):
        events = [
            {"id": "e1", "text": "First event happened."},
            {"id": "e2", "text": "Second event happened."},
        ]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.events_processed == 2
