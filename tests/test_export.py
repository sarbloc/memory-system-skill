"""Tests for export and import logic."""

import io
import json

import pytest

from entity_memory.export import (
    export_json,
    export_markdown,
    import_json,
    reject_future_dated_facts,
)
from entity_memory.models import Entity, Fact


def _entity():
    return Entity(
        id="person:alice",
        type="person",
        facts=[
            Fact(text="Manages auth team", added="2026-02-15", source="event:001",
                 last_seen="2026-03-10", hit_count=4),
            Fact(text="Prefers Slack", added="2026-03-10", source="event:002",
                 last_seen="2026-03-10", hit_count=2, expires="2026-12-31"),
        ],
        last_updated="2026-03-10T14:30:00",
    )


class TestExportJson:
    def test_valid_json(self):
        out = io.StringIO()
        export_json([_entity()], out)
        data = json.loads(out.getvalue())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_entity_fields(self):
        out = io.StringIO()
        export_json([_entity()], out)
        data = json.loads(out.getvalue())[0]
        assert data["id"] == "person:alice"
        assert data["type"] == "person"
        assert len(data["facts"]) == 2

    def test_fact_fields(self):
        out = io.StringIO()
        export_json([_entity()], out)
        fact = json.loads(out.getvalue())[0]["facts"][0]
        assert fact["text"] == "Manages auth team"
        assert fact["hit_count"] == 4
        assert fact["added"] == "2026-02-15"

    def test_empty_export(self):
        out = io.StringIO()
        export_json([], out)
        assert json.loads(out.getvalue()) == []


class TestExportMarkdown:
    def test_has_header(self):
        out = io.StringIO()
        export_markdown([_entity()], out)
        text = out.getvalue()
        assert "## person:alice" in text

    def test_has_facts(self):
        out = io.StringIO()
        export_markdown([_entity()], out)
        text = out.getvalue()
        assert "Manages auth team" in text
        assert "x4" in text

    def test_shows_expires(self):
        out = io.StringIO()
        export_markdown([_entity()], out)
        text = out.getvalue()
        assert "[expires 2026-12-31]" in text


class TestImportJson:
    def test_roundtrip(self):
        out = io.StringIO()
        export_json([_entity()], out)
        data = json.loads(out.getvalue())
        imported = import_json(data)
        assert len(imported) == 1
        assert imported[0].id == "person:alice"
        assert len(imported[0].facts) == 2

    def test_preserves_fact_details(self):
        out = io.StringIO()
        export_json([_entity()], out)
        data = json.loads(out.getvalue())
        imported = import_json(data)
        fact = imported[0].facts[0]
        assert fact.text == "Manages auth team"
        assert fact.hit_count == 4
        assert fact.source == "event:001"

    def test_empty_import(self):
        assert import_json([]) == []


def _temporal_entity():
    """Entity carrying a superseded fact and a current backdated replacement."""
    return Entity(
        id="person:alice",
        type="person",
        facts=[
            Fact(text="lives in London", added="2026-01-01", source="event:001",
                 last_seen="2026-01-01", superseded_at="2026-03-01",
                 superseded_by="lives in Berlin"),
            Fact(text="lives in Berlin", added="2026-03-01", source="event:002",
                 last_seen="2026-03-01", valid_from="2026-03-01"),
        ],
        last_updated="2026-03-01T00:00:00",
    )


class TestExportImportTemporal:
    """Bi-temporal fields must survive a backup round-trip (issue #21)."""

    def test_export_includes_temporal_fields(self):
        out = io.StringIO()
        export_json([_temporal_entity()], out)
        london = json.loads(out.getvalue())[0]["facts"][0]
        assert london["superseded_at"] == "2026-03-01"
        assert london["superseded_by"] == "lives in Berlin"

    def test_roundtrip_preserves_history(self):
        out = io.StringIO()
        export_json([_temporal_entity()], out)
        imported = import_json(json.loads(out.getvalue()))
        by_text = {f.text: f for f in imported[0].facts}
        # The superseded fact must come back superseded, not silently current.
        assert by_text["lives in London"].superseded_at == "2026-03-01"
        assert by_text["lives in London"].superseded_by == "lives in Berlin"
        assert by_text["lives in London"].is_current is False
        assert by_text["lives in Berlin"].valid_from == "2026-03-01"
        assert by_text["lives in Berlin"].is_current is True

    def test_import_pre_issue21_backup(self):
        # A backup written before #21 has no temporal keys; it must still import,
        # with the new fields defaulting to None (fact treated as current).
        legacy = [{
            "id": "person:bob", "type": "person", "last_updated": "",
            "facts": [{"text": "x", "added": "2026-01-01", "source": "e",
                       "last_seen": "2026-01-01", "hit_count": 1}],
        }]
        fact = import_json(legacy)[0].facts[0]
        assert fact.superseded_at is None
        assert fact.valid_from is None
        assert fact.is_current is True

    def test_markdown_marks_superseded(self):
        out = io.StringIO()
        export_markdown([_temporal_entity()], out)
        assert "[superseded 2026-03-01]" in out.getvalue()


class TestRejectFutureDatedFacts:
    """Import must not let a future-dated fact into the current view (issue #24)."""

    def test_future_valid_from_raises(self):
        ents = [Entity(id="person:alice", type="person", facts=[
            Fact(text="moves to Mars", added="2026-01-01", source="e",
                 valid_from="2099-01-01"),
        ])]
        with pytest.raises(ValueError):
            reject_future_dated_facts(ents, today="2026-06-03")

    def test_future_superseded_at_raises(self):
        # A future valid-time END would hide a still-true fact from the current
        # view — same corruption as a future start, so it's rejected too.
        ents = [Entity(id="person:alice", type="person", facts=[
            Fact(text="lives in London", added="2026-01-01", source="e",
                 superseded_at="2099-01-01", superseded_by="lives on Mars"),
        ])]
        with pytest.raises(ValueError):
            reject_future_dated_facts(ents, today="2026-06-03")

    def test_past_same_day_and_none_pass(self):
        # Backdated, same-day, and absent dates are all fine — only a valid-time
        # date strictly after today is rejected.
        ents = [Entity(id="person:alice", type="person", facts=[
            Fact(text="backdated", added="2026-06-03", source="e",
                 valid_from="2026-01-01"),
            Fact(text="same day", added="2026-06-03", source="e",
                 valid_from="2026-06-03"),
            Fact(text="past supersession", added="2026-06-03", source="e",
                 superseded_at="2026-06-03", superseded_by="newer"),
            Fact(text="no valid_from", added="2026-06-03", source="e"),
        ])]
        reject_future_dated_facts(ents, today="2026-06-03")  # must not raise
