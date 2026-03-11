"""CLI entry point for the memory command."""

from datetime import datetime

import click

from entity_memory.client import (
    collection_stats,
    delete_entity,
    ensure_collections,
    get_client,
    get_entity,
    scroll_entities,
    upsert_entity,
)
from entity_memory.merge import build_search_text, merge
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


if __name__ == "__main__":
    main()
