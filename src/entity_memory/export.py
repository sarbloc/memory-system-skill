"""Export entities to JSON/markdown and import from JSON."""

from __future__ import annotations

import json
from typing import TextIO

from entity_memory.models import Entity, Fact


def export_json(entities: list[Entity], out: TextIO) -> None:
    """Export entities as JSON array (no embeddings)."""
    data = []
    for e in entities:
        data.append({
            "id": e.id,
            "type": e.type,
            "last_updated": e.last_updated,
            "facts": [
                {
                    "text": f.text,
                    "added": f.added,
                    "source": f.source,
                    "expires": f.expires,
                    "last_seen": f.last_seen,
                    "hit_count": f.hit_count,
                }
                for f in e.facts
            ],
        })
    json.dump(data, out, indent=2)
    out.write("\n")


def export_markdown(entities: list[Entity], out: TextIO) -> None:
    """Export entities as human-readable markdown."""
    for e in entities:
        out.write(f"## {e.id}\n")
        for f in e.facts:
            expires = f" [expires {f.expires}]" if f.expires else ""
            out.write(f"- {f.text} (x{f.hit_count}, since {f.added}){expires}\n")
        out.write("\n")


def import_json(data: list[dict]) -> list[Entity]:
    """Parse a JSON export back into Entity objects."""
    entities = []
    for item in data:
        facts = [
            Fact(
                text=f["text"],
                added=f["added"],
                source=f["source"],
                expires=f.get("expires"),
                last_seen=f.get("last_seen"),
                hit_count=f.get("hit_count", 1),
            )
            for f in item.get("facts", [])
        ]
        entities.append(Entity(
            id=item["id"],
            type=item["type"],
            facts=facts,
            last_updated=item.get("last_updated", ""),
        ))
    return entities
