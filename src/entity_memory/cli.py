"""CLI entry point for the memory command."""

import uuid
from datetime import datetime, timedelta

import click
from qdrant_client.models import FieldCondition, Filter, MatchValue

from entity_memory.client import (
    DOMAINS,
    collection_name,
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    scroll_entities,
    store_event,
    upsert_entity,
)
from entity_memory.extract import MATCH_THRESHOLD
from entity_memory.merge import build_search_text, compact, drop_expired, mark_superseded, merge
from entity_memory.models import Entity, Fact
from entity_memory.pipeline import DEFAULT_BATCH_SIZE, run_extraction


def _get_embedder():
    """Lazy-load the embedder (avoids slow import on every CLI call)."""
    from entity_memory.embedder import Embedder

    return Embedder()


@click.group()
def main():
    """Entity memory system for OpenClaw agents."""
    pass


@main.command()
def init():
    """Create collections and indexes if they don't exist."""
    client = get_client()
    created = ensure_collections(client)
    if created:
        click.echo(f"Created collections: {', '.join(created)}")
    else:
        click.echo("All collections already exist.")


@main.command()
def stats():
    """Show collection sizes and health."""
    client = get_client()
    st = collection_stats(client)
    for name, info in st.items():
        if info.get("exists") is False:
            click.echo(f"{name}: not created")
        else:
            click.echo(f"{name}: {info['points']} points")


@main.command()
@click.option("--type", "entity_type", required=True, help="Entity type: person|project|tool|preference|decision")
@click.option("--id", "entity_id", required=True, help="Entity key (e.g. alice, dashboard)")
@click.option("--content", required=True, help="Fact text to store")
@click.option(
    "--supersedes", default=None,
    help="Text of an existing current fact this replaces. "
         "Marks it superseded (kept for --as-of queries, hidden from search).",
)
@click.option(
    "--valid-from", "valid_from", default=None,
    help="ISO date the new fact became true in the world, if not today.",
)
def store(entity_type: str, entity_id: str, content: str, supersedes: str | None, valid_from: str | None):
    """Directly upsert an entity with a fact."""
    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()

    full_id = f"{entity_type}:{entity_id}" if ":" not in entity_id else entity_id
    now = datetime.utcnow()
    today = now.date().isoformat()

    # Future-effective dating isn't supported yet (issue #24) — see memory_store.
    if valid_from is not None and valid_from[:10] > today:
        raise click.ClickException(
            f"future valid_from {valid_from!r} is not supported yet (today is "
            f"{today}); future-effective dating is tracked in issue #24"
        )

    existing = get_entity(client, full_id)
    if existing is None:
        existing = Entity(id=full_id, type=entity_type)

    superseded = None
    if supersedes:
        # Close the old fact exactly when the replacement starts (its valid_from,
        # falling back to today): a backdated replacement must end the old fact at
        # the same date, or an as-of query in the gap sees both as valid (#21).
        marked = mark_superseded(
            existing, supersedes, by=content, on_date=valid_from or today,
        )
        superseded = marked.text if marked is not None else None

    new_fact = Fact(text=content, added=today, source="cli:store", valid_from=valid_from)
    merged = merge(existing, [new_fact], embedder, now=now)

    search_text = build_search_text(merged)
    vector = embedder.embed(search_text)
    upsert_entity(client, merged, vector)

    current_facts = sum(1 for f in merged.facts if f.is_current)
    click.echo(f"Stored: {full_id} ({current_facts} current facts)")
    if supersedes:
        if superseded is not None:
            click.echo(f"  Superseded: {superseded!r}")
        else:
            click.echo(f"  No current fact matched --supersedes {supersedes!r}")


@main.command()
@click.argument("entity_id")
def get(entity_id: str):
    """Retrieve a single entity's full detail."""
    client = get_client()
    entity = get_entity(client, entity_id)
    if entity is None:
        click.echo(f"Not found: {entity_id}")
        raise SystemExit(1)

    click.echo(f"{entity.type}:{entity.id}")
    click.echo(f"Updated: {entity.last_updated}")
    click.echo(f"Facts ({len(entity.facts)}):")
    for f in entity.facts:
        expires = f" [expires {f.expires}]" if f.expires else ""
        superseded = f" [superseded {f.superseded_at}]" if f.superseded_at else ""
        click.echo(f"  [{f.hit_count}x] {f.text} (since {f.added}, last seen {f.last_seen}){expires}{superseded}")


@main.command()
@click.argument("entity_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete(entity_id: str, yes: bool):
    """Delete an entity permanently."""
    if not yes:
        click.confirm(f"Delete {entity_id}?", abort=True)

    client = get_client()
    if delete_entity(client, entity_id):
        click.echo(f"Deleted: {entity_id}")
    else:
        click.echo(f"Not found: {entity_id}")
        raise SystemExit(1)


@main.command(name="list")
@click.option("--type", "entity_type", default=None, help="Filter by type")
def list_entities(entity_type: str | None):
    """List all known entities."""
    client = get_client()
    entities = scroll_entities(client, entity_type)
    if not entities:
        click.echo("No entities found.")
        return
    for e in entities:
        updated = e.last_updated[:10] if e.last_updated else "unknown"
        click.echo(f"{e.id} ({len(e.facts)} facts, updated {updated})")


@main.command()
@click.argument("query")
@click.option("--type", "entity_type", default=None, help="Filter by entity type")
@click.option("--limit", default=5, help="Max results to return")
@click.option(
    "--as-of", "as_of", default=None,
    help="ISO date: show each entity as it was then (superseded facts in effect "
         "at that date are shown). Default hides superseded facts.",
)
def search(query: str, entity_type: str | None, limit: int, as_of: str | None):
    """Semantic search across all entities."""
    from entity_memory.search import search_entities, format_results

    client = get_client()
    embedder = _get_embedder()
    results = search_entities(
        client, query, embedder, entity_type=entity_type, limit=limit, as_of=as_of,
    )
    click.echo(format_results(results))


EVENT_TTL_DAYS = 30


@main.command()
@click.argument("text")
@click.option("--source", default="conversation", help="Event source: conversation|task|cron")
@click.option("--agent", default="main", help="Agent name")
def event(text: str, source: str, agent: str):
    """Log a raw observation. No entity extraction happens immediately."""
    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()

    now = datetime.utcnow()
    event_id = str(uuid.uuid4())
    vector = embedder.embed(text)
    expires = (now + timedelta(days=EVENT_TTL_DAYS)).isoformat()

    store_event(
        client, event_id=event_id, text=text, vector=vector,
        timestamp=now.isoformat(), source=source, agent=agent, expires=expires,
    )
    click.echo(f"Logged event: {event_id[:8]}...")


@main.command()
@click.option("--since", default=None, help="Process events from last N minutes (e.g. 55m)")
@click.option("--all", "process_all", is_flag=True, help="Process all unextracted events")
@click.option(
    "--domain", type=click.Choice(DOMAINS), default="shared",
    help="Domain to extract within: shared|dev|personal",
)
@click.option(
    "--batch", "batch_size", default=DEFAULT_BATCH_SIZE, show_default=True,
    help="Events per batch; each batch commits and reports progress before the next.",
)
@click.option(
    "--threshold", default=MATCH_THRESHOLD, show_default=True, type=float,
    help="Cosine gate for a sentence→fact match. Lower = enrich more aggressively.",
)
def extract(
    since: str | None, process_all: bool, domain: str, batch_size: int, threshold: float
):
    """Process unextracted events into entity upserts.

    Events are processed in batches so a large backlog makes visible,
    kill-safe progress. Only events that enriched an existing entity are
    marked extracted; fully-unmatched events stay as new-entity candidates.
    """
    client = get_client()
    embedder = _get_embedder()

    since_dt = None
    if since and not process_all:
        minutes = int(since.rstrip("m"))
        since_dt = datetime.utcnow() - timedelta(minutes=minutes)

    def _progress(info: dict) -> None:
        click.echo(
            f"  batch {info['batch']}/{info['total_batches']}: "
            f"{info['events_in_batch']} events, "
            f"{info['matched_in_batch']} enriched "
            f"({info['cumulative_events']}/{info['total_events']} done)"
        )

    summary = run_extraction(
        client, embedder, domain=domain, since=since_dt,
        batch_size=batch_size, threshold=threshold, progress=_progress,
    )

    if summary["events_processed"] == 0:
        click.echo("No unextracted events found.")
        return

    click.echo(f"Processed {summary['events_processed']} events:")
    click.echo(f"  {summary['matched']} sentences matched to entities")
    click.echo(f"  {summary['unmatched']} unmatched sentences")
    for s in summary["unmatched_samples"]:
        click.echo(f"  UNMATCHED: \"{s}\"")


@main.command(name="dismiss")
@click.option(
    "--source", "src", required=True,
    help="Only dismiss unextracted events whose 'source' field equals this "
         "(e.g. claude_ai_import_conversation).",
)
@click.option("--domain", default="shared", help="Domain: shared|dev|personal")
@click.option(
    "--keep", "keep", multiple=True,
    help="Event ID to exclude from dismissal. Repeatable.",
)
@click.option(
    "--apply", "apply_changes", is_flag=True,
    help="Actually mark matched events extracted. Without this flag the command "
         "is a dry-run (prints count + sample, writes nothing).",
)
def dismiss(src: str, domain: str, keep: tuple[str, ...], apply_changes: bool):
    """Bulk-dismiss unextracted events by source, marking them extracted.

    Built for one-time cleanup of bulk imports (e.g. the Claude.ai conversation
    import) whose events are not memory-worthy: it marks them extracted=True
    with no entity created, so the extractor stops reconsidering them. Events
    are NOT deleted — their existing 30-day TTL still applies. Dry-run by
    default; pass --apply to commit.
    """
    client = get_client()
    coll = collection_name(domain, "events")
    if not client.collection_exists(coll):
        click.echo(f"No events collection for domain {domain!r}.")
        raise SystemExit(1)

    keep_set = set(keep)
    scroll_filter = Filter(
        must=[
            FieldCondition(key="extracted", match=MatchValue(value=False)),
            FieldCondition(key="source", match=MatchValue(value=src)),
        ]
    )

    matched_ids: list = []
    sample: list[tuple[str, str]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=coll, scroll_filter=scroll_filter,
            limit=256, offset=offset, with_payload=True,
        )
        for p in points:
            if str(p.id) in keep_set:
                continue
            matched_ids.append(p.id)
            if len(sample) < 5:
                text = (p.payload or {}).get("text", "")
                sample.append((str(p.id), text.replace("\n", " ")[:100]))
        if offset is None:
            break

    click.echo(
        f"Matched {len(matched_ids)} unextracted events "
        f"(source={src!r}, domain={domain!r}, {len(keep_set)} excluded)."
    )
    for eid, snippet in sample:
        click.echo(f"  {eid}  {snippet}")
    if len(matched_ids) > len(sample):
        click.echo(f"  ... and {len(matched_ids) - len(sample)} more")

    if not apply_changes:
        click.echo("DRY-RUN: nothing written. Pass --apply to mark these extracted.")
        return

    if not matched_ids:
        click.echo("Nothing to dismiss.")
        return

    batch = 500
    for i in range(0, len(matched_ids), batch):
        chunk = matched_ids[i : i + batch]
        client.set_payload(
            collection_name=coll, payload={"extracted": True}, points=chunk,
            wait=True,
        )
    click.echo(f"Dismissed {len(matched_ids)} events (extracted=True).")


@main.command(name="compact")
@click.option("--max-facts", default=20, help="Max facts per entity")
def compact_cmd(max_facts: int):
    """Run compaction on all entities exceeding the facts limit."""
    client = get_client()
    embedder = _get_embedder()
    entities = scroll_entities(client)

    compacted = 0
    now = datetime.utcnow()
    for entity in entities:
        if len(entity.facts) > max_facts:
            entity = compact(entity, max_facts=max_facts, now=now)
            vector = embedder.embed(build_search_text(entity))
            upsert_entity(client, entity, vector)
            compacted += 1

    click.echo(f"Compacted {compacted} entities (max_facts={max_facts}).")


@main.command()
def expire():
    """Garbage-collect expired events and entity facts."""
    client = get_client()
    embedder = _get_embedder()
    now = datetime.utcnow()

    # Expire events
    expired_events = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name="events", limit=100, offset=offset, with_payload=True,
        )
        expired_ids = []
        for p in points:
            expires = p.payload.get("expires")
            if expires and datetime.fromisoformat(expires) < now:
                expired_ids.append(p.id)
        if expired_ids:
            client.delete(collection_name="events", points_selector=expired_ids)
            expired_events += len(expired_ids)
        if next_offset is None:
            break
        offset = next_offset

    # Expire entity facts
    updated_entities = 0
    entities = scroll_entities(client)
    for entity in entities:
        before = len(entity.facts)
        entity.facts = drop_expired(entity.facts, now)
        if len(entity.facts) < before:
            entity.last_updated = now.isoformat()
            vector = embedder.embed(build_search_text(entity))
            upsert_entity(client, entity, vector)
            updated_entities += 1

    click.echo(f"Expired {expired_events} events, updated {updated_entities} entities.")


@main.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", help="Output format")
def export_cmd(fmt: str):
    """Export all entities for backup or inspection."""
    import sys
    from entity_memory.export import export_json, export_markdown

    client = get_client()
    entities = scroll_entities(client)

    if fmt == "json":
        export_json(entities, sys.stdout)
    else:
        export_markdown(entities, sys.stdout)


@main.command(name="import")
@click.argument("file", type=click.Path(exists=True))
def import_cmd(file: str):
    """Import entities from a JSON export. Runs merge logic (won't duplicate)."""
    import json as json_mod
    from entity_memory.export import import_json, reject_future_valid_from

    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()
    now = datetime.utcnow()

    with open(file) as f:
        data = json_mod.load(f)

    imported = import_json(data)
    # Validate the whole backup before writing anything, so a bad fact aborts the
    # import atomically rather than half-applying it (issue #24).
    try:
        reject_future_valid_from(imported, now.date().isoformat())
    except ValueError as exc:
        raise click.ClickException(str(exc))

    for entity in imported:
        existing = get_entity(client, entity.id)
        if existing is None:
            existing = Entity(id=entity.id, type=entity.type)

        merged = merge(existing, entity.facts, embedder, now=now)
        vector = embedder.embed(build_search_text(merged))
        upsert_entity(client, merged, vector)

    click.echo(f"Imported {len(imported)} entities.")


@main.command()
def mcp():
    """Run the MCP stdio server (used by Endurance and other MCP clients)."""
    from entity_memory.mcp_server import run

    run()


if __name__ == "__main__":
    main()
