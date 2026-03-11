# Entity Memory

Persistent entity-based memory system for OpenClaw agents. Stores knowledge as discrete entities that are upserted in place — growth is proportional to unique concepts, not time.

## Quick start

```bash
# Start Qdrant
docker compose up -d

# Install
pip install -e .

# Initialize collections
memory init

# Store a fact
memory store --type person --id alice --content "Manages the auth team"

# Search
memory search "who handles authentication"

# Log an observation
memory event "Task completed: built auth API. Alice reviewed."

# Process events into entities
memory extract --all
```

## Architecture

- **Qdrant** (Docker) — vector database for semantic search
- **all-MiniLM-L6-v2** — local embeddings, 384 dims, no API calls
- **Three collections**: `entities` (long-term knowledge), `events` (30-day rolling window), `decisions` (permanent)

All operations are deterministic (no LLM calls). Deduplication, merging, and compaction use cosine similarity and scoring algorithms.

## Configuration

Config file: `~/.openclaw/memory.json`

```json
{
  "qdrant": {
    "url": "http://127.0.0.1:6333",
    "api_key_env": "QDRANT_API_KEY"
  }
}
```

## OpenClaw skill

Install as a skill:
```bash
bash scripts/install.sh
```

See `skill/SKILL.md` for agent integration instructions.

## Development

```bash
pip install -e .
python3 -m pytest tests/ -v
```

See `MEMORY-SYSTEM-SPEC.md` for the full specification.
