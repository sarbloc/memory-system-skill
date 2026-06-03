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
                    # Bi-temporal fields (issue #21): without these, a backup
                    # round-trip would silently turn superseded history back into
                    # current facts and lose valid-time provenance.
                    "valid_from": f.valid_from,
                    "superseded_at": f.superseded_at,
                    "superseded_by": f.superseded_by,
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
            superseded = f" [superseded {f.superseded_at}]" if f.superseded_at else ""
            out.write(
                f"- {f.text} (x{f.hit_count}, since {f.added}){expires}{superseded}\n"
            )
        out.write("\n")


def reject_future_valid_from(entities: list[Entity], today: str) -> None:
    """Raise ``ValueError`` if any fact is dated to take effect after ``today``.

    Future-effective dating isn't supported yet (issue #24): the default
    ``is_current`` view treats a fact as live the moment it isn't superseded,
    which only holds for backdated/same-day facts. Import enforces the same
    invariant the store path guards at write time, so a hand-edited backup can't
    sneak a not-yet-effective fact into the current search view.
    """
    for e in entities:
        for f in e.facts:
            if f.valid_from is not None and f.valid_from[:10] > today:
                raise ValueError(
                    f"backup has a future valid_from {f.valid_from!r} on {e.id} "
                    f"(today is {today}); future-effective dating is not supported "
                    f"yet (issue #24)"
                )


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
                # .get() so pre-#21 backups (without these keys) still import.
                valid_from=f.get("valid_from"),
                superseded_at=f.get("superseded_at"),
                superseded_by=f.get("superseded_by"),
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
