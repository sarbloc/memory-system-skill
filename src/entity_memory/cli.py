"""CLI entry point for the memory command."""

import click

from entity_memory.client import get_client, ensure_collections, collection_stats


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


if __name__ == "__main__":
    main()
