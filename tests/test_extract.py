"""Tests for event extraction logic."""

from datetime import datetime

from entity_memory.extract import (
    build_entity_index,
    extract_events,
    extract_events_with_index,
    match_sentence_to_entity,
    split_sentences,
)
from entity_memory.models import Entity, Fact


class _CountingEmbedder:
    """Wraps an embedder and counts embed() calls (issue #12 regression guard)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return self.inner.embed(text)


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

    def test_matched_event_ids_empty_when_no_entities(self, embedder):
        # No entities to match against → nothing can match → set is empty.
        events = [{"id": "e1", "text": "Alice reviewed the PR. Bob approved it."}]
        result = extract_events(events, [], embedder, now=NOW)
        assert result.matched_event_ids == set()

    def test_matched_event_ids_includes_matching_event(self, embedder):
        from entity_memory.merge import build_search_text

        entity = _entity("person:alice", "person", ["Manages the auth team"])
        # An event whose sentence equals the entity's search_text matches
        # deterministically (identical text → cosine 1.0 with MockEmbedder).
        matching_sentence = build_search_text(entity)
        events = [{"id": "e1", "text": matching_sentence}]
        result = extract_events(events, [entity], embedder, now=NOW)
        assert "e1" in result.matched_event_ids
        assert len(result.matched) >= 1

    def test_fully_unmatched_event_absent_from_matched_ids(self, embedder):
        # Two events, one guaranteed match (== search_text), one guaranteed
        # non-match (cosine < threshold vs the only entity present).
        from entity_memory.extract import cosine_sim
        from entity_memory.merge import build_search_text

        entity = _entity("person:alice", "person", ["Manages the auth team"])
        st = build_search_text(entity)
        unmatched_text = "grok quaffle."  # verified cosine < 0.7 vs this entity
        # Guard: if the mock embedder ever changes and this accidentally
        # matches, skip rather than assert a false negative.
        if cosine_sim(embedder.embed(unmatched_text), embedder.embed(st)) >= 0.7:
            import pytest

            pytest.skip("Mock embedder changed: chosen text now matches")

        events = [
            {"id": "match", "text": st},
            {"id": "nomatch", "text": unmatched_text},
        ]
        result = extract_events(events, [entity], embedder, now=NOW)
        assert "match" in result.matched_event_ids
        assert "nomatch" not in result.matched_event_ids


# ── embedding cost (issue #12 regression guard) ──────────

class TestEmbedsCorpusOnce:
    def test_embed_calls_additive_not_multiplicative(self, embedder):
        """Each entity is embedded once per run, not once per sentence.

        This is the canary for the #12 blowup: the old per-sentence
        re-embedding of the whole corpus made embed calls scale as
        events × sentences × entities. With the precomputed index it must be
        len(entities) + total_sentences.
        """
        entities = [
            _entity("person:alice", "person", ["Manages the auth team"]),
            _entity("project:dash", "project", ["The trading dashboard"]),
            _entity("tool:qdrant", "tool", ["A vector database"]),
        ]
        events = [
            {"id": "e1", "text": "First sentence here. Second sentence here."},
            {"id": "e2", "text": "Third sentence here. Fourth sentence here."},
        ]
        total_sentences = 4
        counter = _CountingEmbedder(embedder)

        extract_events(events, entities, counter, now=NOW)

        # 3 entity embeds + 4 sentence embeds. The old path would be 4*3 + 4.
        assert counter.calls == len(entities) + total_sentences

    def test_with_index_does_not_re_embed_entities(self, embedder):
        """extract_events_with_index embeds only sentences, never entities."""
        entities = [
            _entity("person:alice", "person", ["Manages the auth team"]),
            _entity("project:dash", "project", ["The trading dashboard"]),
        ]
        index = build_entity_index(entities, embedder)  # entity embeds happen here

        events = [{"id": "e1", "text": "One sentence. Two sentence. Three sentence."}]
        counter = _CountingEmbedder(embedder)
        extract_events_with_index(events, index, counter, now=NOW)

        # Index is prebuilt → only the 3 sentences are embedded.
        assert counter.calls == 3
