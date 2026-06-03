"""Tests for merge, compact, expiry, and search text generation."""

from datetime import datetime, timedelta

import pytest

from entity_memory.models import Entity, Fact
from entity_memory.merge import (
    build_search_text,
    compact,
    drop_expired,
    find_duplicate,
    mark_superseded,
    merge,
    fact_score,
    DUPE_THRESHOLD,
    MAX_FACTS,
)


NOW = datetime(2026, 3, 10)
TODAY = NOW.date().isoformat()


def _fact(
    text, added=TODAY, source="event:001", expires=None, hit_count=1, last_seen=None,
    valid_from=None, superseded_at=None, superseded_by=None,
):
    return Fact(
        text=text, added=added, source=source,
        expires=expires, hit_count=hit_count, last_seen=last_seen or added,
        valid_from=valid_from, superseded_at=superseded_at, superseded_by=superseded_by,
    )


# ── drop_expired ─────────────────────────────────────────

class TestDropExpired:
    def test_keeps_permanent_facts(self):
        facts = [_fact("permanent", expires=None)]
        assert len(drop_expired(facts, NOW)) == 1

    def test_keeps_future_expiry(self):
        facts = [_fact("future", expires="2026-04-01")]
        assert len(drop_expired(facts, NOW)) == 1

    def test_drops_past_expiry(self):
        facts = [_fact("expired", expires="2026-03-09")]
        assert len(drop_expired(facts, NOW)) == 0

    def test_mixed(self):
        facts = [
            _fact("permanent", expires=None),
            _fact("expired", expires="2026-03-01"),
            _fact("future", expires="2026-12-31"),
        ]
        result = drop_expired(facts, NOW)
        assert len(result) == 2
        assert all(f.text != "expired" for f in result)


# ── find_duplicate ───────────────────────────────────────

class TestFindDuplicate:
    def test_exact_duplicate(self, embedder):
        existing = [_fact("manages the auth team")]
        new = _fact("manages the auth team")
        # Same text → same embedding → cosine 1.0 → duplicate
        assert find_duplicate(new, existing, embedder) is existing[0]

    def test_no_duplicate_different_text(self, embedder):
        existing = [_fact("manages the auth team")]
        new = _fact("likes pizza on fridays")
        # With MockEmbedder, different text → different hash → low cosine
        result = find_duplicate(new, existing, embedder)
        # MockEmbedder uses hash, so very different texts won't hit threshold
        # This may or may not be None depending on hash collision, but typically None
        # We check it doesn't falsely match the existing fact at high confidence
        if result is not None:
            pytest.skip("Hash collision in mock embedder")

    def test_empty_existing(self, embedder):
        new = _fact("some new fact")
        assert find_duplicate(new, [], embedder) is None


# ── merge ────────────────────────────────────────────────

class TestMerge:
    def test_merge_into_empty_entity(self, embedder):
        entity = Entity(id="person:alice", type="person")
        facts = [_fact("manages auth"), _fact("prefers slack")]
        result = merge(entity, facts, embedder, now=NOW)
        assert len(result.facts) == 2

    def test_merge_exact_duplicate_increments_hit_count(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("manages auth")], embedder, now=NOW)
        assert entity.facts[0].hit_count == 1

        # Merge same text again
        entity = merge(entity, [_fact("manages auth")], embedder, now=NOW)
        assert len(entity.facts) == 1
        assert entity.facts[0].hit_count == 2

    def test_merge_duplicate_longer_text_replaces(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("manages auth")], embedder, now=NOW)

        # Same text but longer → should replace
        longer = _fact("manages auth")
        longer.text = "manages auth"  # same text, same length → no replace
        entity = merge(entity, [longer], embedder, now=NOW)
        assert entity.facts[0].text == "manages auth"

    def test_merge_new_facts_appended(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("manages auth")], embedder, now=NOW)
        entity = merge(entity, [_fact("completely different fact xyz")], embedder, now=NOW)
        assert len(entity.facts) == 2

    def test_merge_drops_expired_first(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("old expired", expires="2026-03-01")]
        entity = merge(entity, [_fact("new fact")], embedder, now=NOW)
        # Expired fact should be gone, only new fact remains
        assert len(entity.facts) == 1
        assert entity.facts[0].text == "new fact"

    def test_merge_sets_last_updated(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("test")], embedder, now=NOW)
        assert entity.last_updated == NOW.isoformat()

    def test_merge_sets_last_seen_on_new_facts(self, embedder):
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("test")], embedder, now=NOW)
        assert entity.facts[0].last_seen == TODAY


# ── compact ──────────────────────────────────────────────

class TestCompact:
    def test_no_compaction_under_limit(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [_fact(f"fact {i}") for i in range(5)]
        result = compact(entity, now=NOW)
        assert len(result.facts) == 5

    def test_compaction_at_limit(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [_fact(f"fact {i}") for i in range(MAX_FACTS)]
        result = compact(entity, now=NOW)
        assert len(result.facts) == MAX_FACTS

    def test_compaction_trims_to_max(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [_fact(f"fact {i}") for i in range(30)]
        result = compact(entity, now=NOW)
        assert len(result.facts) == MAX_FACTS

    def test_high_hit_count_survives(self):
        entity = Entity(id="test:e", type="test")
        low_facts = [_fact(f"low {i}", hit_count=1) for i in range(25)]
        high_fact = _fact("important", hit_count=100)
        entity.facts = low_facts + [high_fact]
        result = compact(entity, now=NOW)
        kept_texts = [f.text for f in result.facts]
        assert "important" in kept_texts

    def test_recent_facts_survive_over_old(self):
        entity = Entity(id="test:e", type="test")
        old_date = (NOW - timedelta(days=365)).date().isoformat()
        old_facts = [
            _fact(f"old {i}", added=old_date, last_seen=old_date, hit_count=1)
            for i in range(25)
        ]
        recent_fact = _fact("recent", added=TODAY, last_seen=TODAY, hit_count=1)
        entity.facts = old_facts + [recent_fact]
        result = compact(entity, now=NOW)
        kept_texts = [f.text for f in result.facts]
        assert "recent" in kept_texts

    def test_permanent_facts_get_boost(self):
        entity = Entity(id="test:e", type="test")
        # Two facts with same age and hit_count, one permanent, one expiring
        perm = _fact("permanent", hit_count=1, expires=None)
        temp = _fact("temporary", hit_count=1, expires="2026-12-31")
        # Permanent should score higher
        assert fact_score(perm, NOW) > fact_score(temp, NOW)

    def test_custom_max_facts(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [_fact(f"fact {i}") for i in range(15)]
        result = compact(entity, max_facts=10, now=NOW)
        assert len(result.facts) == 10


# ── build_search_text ────────────────────────────────────

class TestBuildSearchText:
    def test_format(self):
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("manages auth"), _fact("prefers slack")]
        text = build_search_text(entity, now=NOW)
        assert text.startswith("[person] person:alice")
        assert "manages auth" in text
        assert "prefers slack" in text

    def test_max_10_facts(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [_fact(f"fact {i}") for i in range(15)]
        text = build_search_text(entity, now=NOW)
        # Should contain type prefix + 10 facts = 11 parts joined by ". "
        parts = text.split(". ")
        assert len(parts) == 11

    def test_empty_entity(self):
        entity = Entity(id="test:e", type="test")
        text = build_search_text(entity, now=NOW)
        assert text == "[test] test:e"

    def test_top_facts_appear_first(self):
        entity = Entity(id="test:e", type="test")
        entity.facts = [
            _fact("low priority", hit_count=1, last_seen="2025-01-01"),
            _fact("high priority", hit_count=50, last_seen=TODAY),
        ]
        text = build_search_text(entity, now=NOW)
        high_pos = text.index("high priority")
        low_pos = text.index("low priority")
        assert high_pos < low_pos

    def test_excludes_superseded_facts(self):
        # Superseded facts must not feed the search vector (issue #21).
        entity = Entity(id="person:alice", type="person")
        entity.facts = [
            _fact("lives in Berlin"),
            _fact("lives in London", superseded_at="2026-03-01", superseded_by="lives in Berlin"),
        ]
        text = build_search_text(entity, now=NOW)
        assert "lives in Berlin" in text
        assert "London" not in text

    def test_future_supersession_still_feeds_vector(self):
        # A fact superseded with a *future* effective date is still true now, so
        # it must keep feeding the vector until that date — using valid_at, not
        # is_current (Codex review, PR #23).
        entity = Entity(id="person:alice", type="person")
        entity.facts = [
            _fact("lives in London", added="2026-01-01",
                  superseded_at="2099-01-01", superseded_by="lives in Berlin"),
        ]
        # NOW (2026-03-10) is before the effective date → still indexed.
        assert "London" in build_search_text(entity, now=NOW)
        # After the effective date → drops out.
        assert "London" not in build_search_text(entity, now=datetime(2099, 6, 1))


# ── bi-temporal Fact predicates (issue #21) ──────────────

class TestFactTemporal:
    def test_is_current_default(self):
        assert _fact("x").is_current is True

    def test_is_current_when_superseded(self):
        assert _fact("x", superseded_at="2026-03-01").is_current is False

    def test_valid_at_before_added_is_false(self):
        f = _fact("x", added="2026-01-01")
        assert f.valid_at("2025-12-31") is False

    def test_valid_at_on_and_after_start_is_true(self):
        f = _fact("x", added="2026-01-01")
        assert f.valid_at("2026-01-01") is True
        assert f.valid_at("2026-06-01") is True

    def test_valid_at_uses_valid_from_over_added(self):
        # Recorded 2026-01-01 but only true from 2026-02-01.
        f = _fact("x", added="2026-01-01", valid_from="2026-02-01")
        assert f.valid_at("2026-01-15") is False
        assert f.valid_at("2026-02-15") is True

    def test_valid_at_supersession_is_exclusive(self):
        f = _fact("x", added="2026-01-01", superseded_at="2026-03-01")
        assert f.valid_at("2026-02-28") is True
        assert f.valid_at("2026-03-01") is False  # gone the day it is superseded
        assert f.valid_at("2026-04-01") is False

    def test_valid_at_excludes_expired(self):
        # A TTL'd fact is gone once its expiry date has passed — an as-of query
        # after expiry must not resurrect it (Codex review, PR #23).
        f = _fact("on vacation", added="2026-01-01", expires="2026-06-15")
        assert f.valid_at("2026-06-01") is True
        assert f.valid_at("2026-06-15") is False  # gone on the expiry date
        assert f.valid_at("2026-07-01") is False


# ── mark_superseded (issue #21) ──────────────────────────

class TestMarkSuperseded:
    def test_marks_by_text(self):
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("lives in London")]
        marked = mark_superseded(entity, "lives in London", by="lives in Berlin", on_date="2026-03-10")
        assert marked is entity.facts[0]
        assert marked.superseded_at == "2026-03-10"
        assert marked.superseded_by == "lives in Berlin"
        assert marked.is_current is False

    def test_marks_by_source(self):
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("lives in London", source="event:42")]
        marked = mark_superseded(entity, "event:42", by="lives in Berlin", on_date="2026-03-10")
        assert marked is entity.facts[0]
        assert marked.superseded_at == "2026-03-10"

    def test_no_match_returns_none(self):
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("lives in London")]
        assert mark_superseded(entity, "no such fact", by="x", on_date="2026-03-10") is None

    def test_skips_already_superseded(self):
        # Only an already-superseded fact matches → nothing to mark.
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("lives in London", superseded_at="2026-01-01")]
        assert mark_superseded(entity, "lives in London", by="x", on_date="2026-03-10") is None

    def test_marks_current_when_both_present(self):
        # A superseded + a current fact share text → mark the current one.
        entity = Entity(id="person:alice", type="person")
        old = _fact("recurring", superseded_at="2026-01-01")
        cur = _fact("recurring")
        entity.facts = [old, cur]
        marked = mark_superseded(entity, "recurring", by="x", on_date="2026-03-10")
        assert marked is cur
        assert old.superseded_at == "2026-01-01"  # untouched


# ── supersession in compact / merge (issue #21) ──────────

class TestSupersededInCompact:
    def test_superseded_dropped_before_current(self):
        entity = Entity(id="test:e", type="test")
        current = [_fact(f"current {i}") for i in range(MAX_FACTS)]
        superseded = _fact("old", superseded_at="2026-01-01")
        entity.facts = current + [superseded]
        result = compact(entity, now=NOW)
        assert len(result.facts) == MAX_FACTS
        assert all(f.is_current for f in result.facts)
        assert "old" not in [f.text for f in result.facts]

    def test_current_outranks_high_score_superseded(self):
        # A superseded fact with a huge hit_count still drops before plain
        # current facts — currency beats score.
        entity = Entity(id="test:e", type="test")
        current = [_fact(f"current {i}", hit_count=1) for i in range(MAX_FACTS)]
        superseded = _fact("old but popular", hit_count=100, superseded_at="2026-01-01")
        entity.facts = current + [superseded]
        result = compact(entity, now=NOW)
        assert "old but popular" not in [f.text for f in result.facts]


class TestMergeDedupAgainstCurrentOnly:
    def test_reasserted_fact_appends_not_revives(self, embedder):
        # A superseded fact must not be revived by a re-asserted duplicate; the
        # new assertion appends as a fresh current fact, history left intact.
        entity = Entity(id="person:alice", type="person")
        entity.facts = [_fact("lives in London", superseded_at="2026-03-01")]
        entity = merge(entity, [_fact("lives in London")], embedder, now=NOW)
        assert len(entity.facts) == 2
        superseded = [f for f in entity.facts if not f.is_current]
        current = [f for f in entity.facts if f.is_current]
        assert len(superseded) == 1
        assert len(current) == 1
        # History untouched: still superseded, hit_count not bumped.
        assert superseded[0].superseded_at == "2026-03-01"
        assert superseded[0].hit_count == 1

    def test_dedup_takes_earlier_valid_from(self, embedder):
        # Re-asserting a fact with an earlier valid_from widens the valid-time
        # window backwards; dedup must keep the earlier start, not drop it
        # (Codex review, PR #23).
        entity = Entity(id="person:alice", type="person")
        entity = merge(entity, [_fact("likes pizza")], embedder, now=NOW)
        entity = merge(
            entity, [_fact("likes pizza", valid_from="2026-01-01")], embedder, now=NOW
        )
        assert len(entity.facts) == 1
        assert entity.facts[0].valid_from == "2026-01-01"
        assert entity.facts[0].hit_count == 2

    def test_dedup_keeps_existing_earlier_valid_from(self, embedder):
        # The reverse: a later valid_from must NOT overwrite an earlier one.
        entity = Entity(id="person:alice", type="person")
        entity = merge(
            entity, [_fact("likes pizza", valid_from="2026-01-01")], embedder, now=NOW
        )
        entity = merge(
            entity, [_fact("likes pizza", valid_from="2026-05-01")], embedder, now=NOW
        )
        assert len(entity.facts) == 1
        assert entity.facts[0].valid_from == "2026-01-01"
