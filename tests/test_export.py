"""Tests for export and import logic."""

import io
import json

from entity_memory.export import export_json, export_markdown, import_json
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
