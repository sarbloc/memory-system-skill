"""CLI entry point for the memory command."""

import uuid
from datetime import datetime, timedelta

import click

from entity_memory.client import (
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    get_unextracted_events,
    mark_event_extracted,
    scroll_entities,
    store_event,
    upsert_entity,
)
from entity_memory.merge import build_search_text, compact, drop_expired, merge
from entity_memory.models import Entity, Fact


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
def store(entity_type: str, entity_id: str, content: str):
    """Directly upsert an entity with a fact."""
    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()

    full_id = f"{entity_type}:{entity_id}" if ":" not in entity_id else entity_id
    now = datetime.utcnow()
    today = now.date().isoformat()

    existing = get_entity(client, full_id)
    if existing is None:
        existing = Entity(id=full_id, type=entity_type)

    new_fact = Fact(text=content, added=today, source="cli:store")
    merged = merge(existing, [new_fact], embedder, now=now)

    search_text = build_search_text(merged)
    vector = embedder.embed(search_text)
    upsert_entity(client, merged, vector)

    click.echo(f"Stored: {full_id} ({len(merged.facts)} facts)")


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
        click.echo(f"  [{f.hit_count}x] {f.text} (since {f.added}, last seen {f.last_seen}){expires}")


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
def search(query: str, entity_type: str | None, limit: int):
    """Semantic search across all entities."""
    from entity_memory.search import search_entities, format_results

    client = get_client()
    embedder = _get_embedder()
    results = search_entities(client, query, embedder, entity_type=entity_type, limit=limit)
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
def extract(since: str | None, process_all: bool):
    """Process unextracted events into entity upserts."""
    from entity_memory.extract import extract_events

    client = get_client()
    embedder = _get_embedder()

    since_dt = None
    if since and not process_all:
        minutes = int(since.rstrip("m"))
        since_dt = datetime.utcnow() - timedelta(minutes=minutes)

    events = get_unextracted_events(client, since=since_dt)
    if not events:
        click.echo("No unextracted events found.")
        return

    entities = scroll_entities(client)
    now = datetime.utcnow()
    today = now.date().isoformat()

    result = extract_events(events, entities, embedder, now=now)

    # Merge matched sentences into their entities
    for sentence, entity in result.matched:
        new_fact = Fact(text=sentence, added=today, source="extract")
        merged = merge(entity, [new_fact], embedder, now=now)
        vector = embedder.embed(build_search_text(merged))
        upsert_entity(client, merged, vector)

    # Mark events as extracted
    for ev in events:
        mark_event_extracted(client, ev["id"])

    click.echo(f"Processed {result.events_processed} events:")
    click.echo(f"  {len(result.matched)} sentences matched to entities")
    click.echo(f"  {len(result.unmatched)} unmatched sentences")

    for s in result.unmatched:
        click.echo(f"  UNMATCHED: \"{s}\"")


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
    from entity_memory.export import import_json

    client = get_client()
    ensure_collections(client)
    embedder = _get_embedder()
    now = datetime.utcnow()

    with open(file) as f:
        data = json_mod.load(f)

    imported = import_json(data)
    for entity in imported:
        existing = get_entity(client, entity.id)
        if existing is None:
            existing = Entity(id=entity.id, type=entity.type)

        merged = merge(existing, entity.facts, embedder, now=now)
        vector = embedder.embed(build_search_text(merged))
        upsert_entity(client, merged, vector)

    click.echo(f"Imported {len(imported)} entities.")


if __name__ == "__main__":
    main()
