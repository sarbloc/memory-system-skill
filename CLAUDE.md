# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Entity-based persistent memory system for OpenClaw agents. Stores knowledge as discrete entities (upserted in place) rather than appending documents. Growth is proportional to unique concepts, not time. Published as a ClawHub community skill.

The full spec is in `MEMORY-SYSTEM-SPEC.md`.

## Tech Stack

- **Language:** Python
- **Vector DB:** Qdrant (Docker container, ports 6333/6334, localhost-only)
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (384 dims, local, no API calls)
- **CLI framework:** Click
- **Package manager:** pip (pyproject.toml)

## Build & Run Commands

```bash
# Install in dev mode
pip install -e .

# Run CLI
memory <command>

# Start Qdrant
docker compose up -d

# Initialize collections
memory init

# Run tests
pytest
pytest tests/test_merge.py           # unit tests (no Qdrant needed)
pytest tests/test_search.py          # integration tests (Qdrant must be running)
pytest tests/test_merge.py::test_name  # single test
```

## Architecture

Three Qdrant collections:
- **`entities`** — long-term knowledge (person, project, tool, preference, decision). Upserted in place, never appended.
- **`events`** — short-term observations (rolling 30-day window, auto-expires). Source material for entity extraction.
- **`decisions`** — architectural/strategic choices. Same structure as entities but permanent, never expire.

### Key modules (`src/entity_memory/`)

| File | Purpose |
|------|---------|
| `cli.py` | Click CLI entry point for all commands |
| `client.py` | Qdrant client wrapper, collection initialization |
| `embedder.py` | sentence-transformers wrapper |
| `models.py` | Dataclasses: Fact, Entity, Event |
| `merge.py` | Merge, compact, dedup, search text generation |
| `extract.py` | Event → entity extraction (sentence splitting, matching) |
| `search.py` | Vector + text + filter search fusion |
| `export.py` | Export/import (JSON and markdown formats) |

### Core algorithms (all deterministic, no LLM)

- **Merge:** New facts compared by cosine similarity. >= 0.9 = duplicate (increment hit_count, keep longer text). < 0.9 = new fact appended.
- **Compaction:** Triggered when facts > 20. Score = `frequency × recency × permanence`. Keep top 20.
- **Extraction:** Events split into sentences, matched against existing entities (cosine > 0.7). Unmatched sentences logged but NOT auto-created as entities.

### Constants

```python
DUPE_THRESHOLD = 0.9
MAX_FACTS = 20
EVENT_TTL_DAYS = 30
HALF_LIFE_DAYS = 30
```

## Testing

- Unit tests use `MockEmbedder` (deterministic hash-based vectors, 48 dims). Tests merge mechanics, not semantic quality.
- Integration tests require a running Qdrant container. Use `conftest.py` fixtures for temporary collections.
- Config lives at `~/.openclaw/memory.json` (or env vars). Qdrant API key via `QDRANT_API_KEY` env var.
