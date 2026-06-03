"""Extraction pipeline: fetch unextracted events and roll matched sentences
into existing entities, in batches.

This is the single orchestration path shared by the ``memory extract`` CLI
command and the ``memory_extract`` MCP tool. Keeping both callers on one helper
is deliberate: the CLI previously drifted from the MCP path and silently
re-introduced the burn-all bug (issue #14). One code path, one behaviour.

Events are processed in batches so that:
  - progress is visible on bulk runs (an optional per-batch callback), and
  - partial work survives a kill — each batch upserts its matched entities and
    marks ONLY that batch's matched events extracted before the next batch
    starts (issue #12).

The entity index is embedded once for the whole run (``build_entity_index``),
not per batch: this is what removes the ``O(events × sentences × entities)``
embedding blowup. Tradeoff: entities enriched earlier in a run do not influence
matching later in the *same* run — those land on the next run. Flat, predictable
cost over the run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from entity_memory.client import (
    get_unextracted_events,
    mark_event_extracted,
    scroll_entities,
    upsert_entity,
)
from entity_memory.extract import (
    MATCH_THRESHOLD,
    build_entity_index,
    extract_events_with_index,
)
from entity_memory.merge import build_search_text, merge
from entity_memory.models import Fact

DEFAULT_BATCH_SIZE = 200

# Shape returned by run_extraction. Kept stable because the MCP memory_extract
# tool returns it verbatim (an Endurance-facing contract).
ProgressFn = Callable[[dict[str, Any]], None]


def _empty_summary() -> dict[str, Any]:
    return {
        "events_processed": 0,
        "matched": 0,
        "unmatched": 0,
        "matched_events": 0,
        "unmatched_samples": [],
    }


def run_extraction(
    client,
    embedder,
    *,
    domain: str = "shared",
    since: Optional[datetime] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    threshold: float = MATCH_THRESHOLD,
    now: Optional[datetime] = None,
    progress: Optional[ProgressFn] = None,
) -> dict[str, Any]:
    """Fetch unextracted events in ``domain`` and enrich matched entities.

    Args:
        client: Qdrant client.
        embedder: embedding model.
        domain: shared | dev | personal.
        since: only events newer than this (None = whole backlog).
        batch_size: events per batch (>= 1). Each batch commits before the next.
        threshold: cosine gate for a sentence→entity match.
        now: clock override (fact timestamps).
        progress: optional callback invoked once per batch with a dict of
            {batch, total_batches, events_in_batch, matched_in_batch,
             cumulative_events, total_events}.

    Returns a summary dict: events_processed, matched, unmatched,
    matched_events, unmatched_samples. ``matched``/``unmatched`` are sentence
    counts; ``matched_events`` is the number of events with >=1 matched
    sentence; ``unmatched_samples`` is the first 10 unmatched sentences.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    events = get_unextracted_events(client, since=since, domain=domain)
    if not events:
        return _empty_summary()

    now = now or datetime.utcnow()
    today = now.date().isoformat()

    # Embed the entity corpus ONCE for the whole run (see module docstring).
    entities = scroll_entities(client, domain=domain)
    index = build_entity_index(entities, embedder)

    total_events = 0
    total_matched = 0
    total_unmatched = 0
    matched_event_ids: set[str] = set()
    unmatched_samples: list[str] = []
    total_batches = (len(events) + batch_size - 1) // batch_size

    for batch_no, start in enumerate(range(0, len(events), batch_size), start=1):
        batch = events[start : start + batch_size]
        result = extract_events_with_index(
            batch, index, embedder, now=now, threshold=threshold
        )

        # Enrich matched entities. merge() mutates the (shared) entity object in
        # the index, so repeated matches across batches accumulate correctly.
        for sentence, entity in result.matched:
            merged = merge(
                entity, [Fact(text=sentence, added=today, source="extract")],
                embedder, now=now,
            )
            upsert_entity(
                client, merged, embedder.embed(build_search_text(merged)),
                domain=domain,
            )

        # Mark extracted ONLY events that enriched an entity, per batch, so a
        # kill mid-run preserves completed batches. Fully-unmatched events stay
        # unextracted as new-entity candidates.
        for event_id in result.matched_event_ids:
            mark_event_extracted(client, event_id, domain=domain)

        total_events += result.events_processed
        total_matched += len(result.matched)
        total_unmatched += len(result.unmatched)
        matched_event_ids |= result.matched_event_ids
        for s in result.unmatched:
            if len(unmatched_samples) >= 10:
                break
            unmatched_samples.append(s)

        if progress is not None:
            progress(
                {
                    "batch": batch_no,
                    "total_batches": total_batches,
                    "events_in_batch": result.events_processed,
                    "matched_in_batch": len(result.matched_event_ids),
                    "cumulative_events": total_events,
                    "total_events": len(events),
                }
            )

    return {
        "events_processed": total_events,
        "matched": total_matched,
        "unmatched": total_unmatched,
        "matched_events": len(matched_event_ids),
        "unmatched_samples": unmatched_samples,
    }
