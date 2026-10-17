# Entity Memory System — Build Spec

## Overview

A persistent memory system for OpenClaw agents that stores knowledge as discrete entities (upserted in place) rather than appending documents. Growth is proportional to unique concepts, not time.

**Target deployment:** Ubuntu server, Docker for Qdrant, Python CLI, OpenClaw skill.
**Target publication:** ClawHub community skill.

---

## Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   OpenClaw   │       │  memory CLI  │       │    Qdrant    │
│   Agent      │──exec─│  (Python)    │──REST─│   (Docker)   │
│  (Agent)     │       │              │       │  port 6333   │
└──────────────┘       │  - event     │       └──────────────┘
                       │  - store     │
                       │  - search    │       ┌──────────────┐
                       │  - extract   │       │ MiniLM-L6-v2 │
                       │  - compact   │──load─│ (local model)│
                       │  - expire    │       │ ~80MB on disk│
                       │  - export    │       └──────────────┘
                       │  - list      │
                       └──────────────┘
```

All embeddings computed locally. No external API calls for memory operations.
Qdrant runs as a single Docker container (~50MB image, ~100MB RAM).

---

## Data Model

### Three collections in Qdrant

#### 1. `entities` — long-term knowledge (grows sub-linearly)

Each point represents one unique concept. Upserted (updated in place), never appended.

```
Point ID:    deterministic hash of entity id (e.g., hash("person:alice"))
Vector:      dense embedding of build_search_text(entity), 384 dims (MiniLM)
Payload: {
    "entity_id":    "person:alice",
    "type":         "person",                    # person|project|tool|preference|decision
    "search_text":  "[person] person:alice. Manages auth team. Prefers Slack.",
    "facts": [
        {
            "text":      "Manages the auth team",
            "added":     "2026-02-15",
            "source":    "event:a1b2c3",
            "expires":   null,
            "last_seen": "2026-03-10",
            "hit_count": 4
        },
        {
            "text":      "Prefers Slack, especially #auth-team channel",
            "added":     "2026-03-10",
            "source":    "event:d4e5f6",
            "expires":   null,
            "last_seen": "2026-03-10",
            "hit_count": 2
        }
    ],
    "last_updated": "2026-03-10T14:30:00"
}
```

#### 2. `events` — short-term observations (rolling window, auto-expires)

Raw observations from conversations and task completions. Source material for entity extraction.

```
Point ID:    UUID
Vector:      dense embedding of event text, 384 dims
Payload: {
    "text":       "Task completed: built auth API with Express + JWT. Alice reviewed and approved.",
    "timestamp":  "2026-03-10T14:30:00",
    "source":     "conversation|task|cron",
    "agent":      "main",
    "extracted":  false,                         # flipped to true after entity extraction
    "expires":    "2026-04-09T14:30:00"          # 30 days from creation
}
```

#### 3. `decisions` — architectural/strategic choices (few, permanent)

Same structure as entities but semantically distinct. Decisions are permanent and never expire.

```
Point ID:    deterministic hash of decision id
Vector:      dense embedding of decision text, 384 dims
Payload: {
    "entity_id":   "chose-openrouter",
    "type":        "decision",
    "search_text": "[decision] chose-openrouter. Single API key for multi-provider. 5.5% credit fee acceptable.",
    "facts": [
        {
            "text":      "Chose OpenRouter over direct API for multi-provider routing",
            "added":     "2026-03-10",
            "source":    "event:x1y2z3",
            "expires":   null,
            "last_seen": "2026-03-10",
            "hit_count": 1
        },
        {
            "text":      "5.5% credit fee, no per-token markup, single billing dashboard",
            "added":     "2026-03-10",
            "source":    "event:x1y2z3",
            "expires":   null,
            "last_seen": "2026-03-10",
            "hit_count": 1
        }
    ],
    "last_updated": "2026-03-10T15:00:00"
}
```

### Collection configuration

```python
from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType, PayloadSchemaType

# All three collections use the same vector config
VECTOR_CONFIG = VectorParams(size=384, distance=Distance.COSINE)

# Create text indexes for hybrid keyword search
TEXT_INDEX = TextIndexParams(type="text", tokenizer=TokenizerType.WORD, min_token_len=2)

# Payload indexes for filtered queries
# On entities and decisions: index "type", "entity_id", "search_text"
# On events: index "extracted", "source", "agent", "text"
```

---

## Entity Merge Logic

No LLM. All operations are deterministic code + cosine similarity.

### Constants

```python
DUPE_THRESHOLD = 0.9        # cosine similarity above this = same fact
MAX_FACTS = 20              # trigger compaction above this
EVENT_TTL_DAYS = 30         # events auto-expire after this
HALF_LIFE_DAYS = 30         # recency decay for compaction scoring
```

### Merge algorithm

```
INPUT:  existing Entity (from Qdrant), list of new Facts
OUTPUT: merged Entity (to upsert back to Qdrant)

1. Drop expired facts from existing entity (expires < now)
2. For each new fact:
   a. Embed the new fact text (local model)
   b. Find closest existing fact by cosine similarity
   c. If cosine >= 0.9 (duplicate):
      - Increment hit_count on existing fact
      - Update last_seen to today
      - If new text is longer than existing text:
        replace text and embedding (richer version wins)
   d. If cosine < 0.9 (new fact):
      - Append to facts list
3. Update entity.last_updated
4. Rebuild search_text from top 10 facts (ranked by score)
5. Re-embed search_text as entity's dense vector
6. Upsert to Qdrant
```

### Compaction algorithm (no LLM)

```
TRIGGER: entity.facts length > MAX_FACTS
STRATEGY: scored pruning, keep top MAX_FACTS

Score per fact = frequency × recency × permanence
  frequency   = 1 + log(1 + hit_count)        # log scale, diminishing returns
  recency     = 2^(-age_days / HALF_LIFE)      # 30-day half-life decay
  permanence  = 1.2 if expires is None else 1.0 # permanent facts get small boost

Sort by score descending, keep top MAX_FACTS, drop the rest.
```

### Search text generation (no LLM)

```
INPUT:  Entity
OUTPUT: string for embedding

Format: "[{type}] {entity_id}. {fact_1}. {fact_2}. ... {fact_10}"
Facts sorted by score (same formula as compaction), take top 10.
```

---

## CLI Interface

Single Python CLI entry point: `memory`

### Commands

#### `memory event <text>`
Log a raw observation. No entity extraction happens immediately.

```bash
memory event "Task completed: built auth API. Alice reviewed and approved."
```

- Generates UUID
- Embeds text locally
- Stores in `events` collection with extracted=false
- Sets expires = now + EVENT_TTL_DAYS

#### `memory store --type <type> --id <id> --content <text>`
Directly upsert an entity or decision. Used when the agent is confident about what to store.

```bash
memory store --type person --id alice --content "Manages the auth team"
memory store --type decision --id chose-qdrant --content "Entity memory over markdown"
```

- Retrieves existing entity from Qdrant (if any)
- Creates Fact from content
- Runs merge algorithm
- Upserts back to Qdrant

#### `memory search <query> [--type <type>] [--limit <n>]`
Semantic search across all collections (or filtered by type).

```bash
memory search "who handles authentication"
memory search "database choice" --type decision
memory search "alice" --type person --limit 3
```

- Embeds query locally
- Queries Qdrant with dense vector search
- Optionally applies payload filter on `type`
- Also runs text match on `search_text` field for keyword fallback
- Returns results as structured text:
  ```
  [0.94] person:alice — Manages the auth team. Prefers Slack. Approved auth API PR.
  [0.71] project:dashboard — Auth module uses Express + JWT. PostgreSQL backend.
  ```

#### `memory extract [--since <duration>]`
Process unextracted events into entity upserts. This is the consolidation step.

```bash
memory extract --since 55m    # process events from last 55 minutes
memory extract --all           # process all unextracted events
```

- Queries events collection where extracted=false (and within time window)
- For each event, uses a lightweight extraction approach:
  1. Split event text into sentences
  2. For each sentence, search entities collection for a match (cosine > 0.7)
  3. If match found: merge the sentence as a new fact on that entity
  4. If no match: create a candidate (logged but not auto-created)
  5. Mark event as extracted=true

**Note on entity creation:** The extract command does NOT auto-create new entities
from events. It only merges into existing entities. New entities are created
explicitly via `memory store`. This prevents hallucinated entities from noisy events.

The candidates (unmatched sentences) are logged to stdout so the agent (or the
heartbeat cron) can decide whether to create new entities from them:
```
UNMATCHED: "Express + JWT used for auth API" (no matching entity found)
UNMATCHED: "Bob joined the standup call" (no matching entity found)
```

The agent can then choose:
```bash
memory store --type person --id bob --content "Joined standup call on 2026-03-10"
```

#### `memory compact [--max-facts <n>]`
Run compaction on all entities exceeding the facts limit.

```bash
memory compact                 # default MAX_FACTS=20
memory compact --max-facts 15  # more aggressive
```

- Queries all entities from Qdrant
- For each entity with facts > max: run compaction algorithm
- Re-embed and upsert compacted entities

#### `memory expire`
Garbage-collect expired events and expired entity facts.

```bash
memory expire
```

- Delete points from `events` collection where expires < now
- For each entity: drop expired facts, re-embed if changed, upsert

#### `memory list [--type <type>]`
List all known entities (IDs and types, not full content).

```bash
memory list
memory list --type person
memory list --type project
```

- Scroll through Qdrant collection
- Output: `person:alice (5 facts, updated 2026-03-10)`

#### `memory export [--format json|md]`
Export all entities for backup or human inspection.

```bash
memory export --format json > memory-backup.json
memory export --format md > memory-readable.md
```

JSON format: array of entity objects (same as Qdrant payload, minus embeddings).
Markdown format: human-readable, one section per entity:
```markdown
## person:alice
- Manages the auth team (×4, since 2026-02-15)
- Prefers Slack, especially #auth-team channel (×2, since 2026-03-10)
- Approved auth API PR (×1, since 2026-03-10)
```

#### `memory import <file>`
Import entities from a JSON export. Runs merge logic for each (won't duplicate).

```bash
memory import memory-backup.json
```

#### `memory get <entity_id>`
Retrieve a single entity's full detail.

```bash
memory get person:alice
```

Output: all facts with hit_count, added date, last_seen, expiry.

#### `memory delete <entity_id>`
Delete an entity permanently. Requires confirmation.

```bash
memory delete person:alice
```

#### `memory stats`
Show collection sizes and health.

```bash
memory stats
```

Output:
```
entities:  47 points (12 person, 8 project, 15 tool, 6 preference, 6 decision)
events:    23 points (oldest: 2026-02-12, 4 unextracted)
decisions: 6 points
```

#### `memory init`
Create collections and indexes if they don't exist. Idempotent — safe to run repeatedly.

```bash
memory init
```

---

## Quick Reference: When To Use What

| Situation | Command | Why |
|-----------|---------|-----|
| Conversation just ended | `memory event "..."` | Raw log, extracted later by cron |
| Task completed by sub-agent | `memory event "..."` | Same — log the outcome |
| User states a clear preference | `memory store --type preference` | Confident, structured fact |
| Architectural decision made | `memory store --type decision` | Permanent, important |
| About to spawn a sub-agent | `memory search "relevant query"` | Gather context for task description |
| User asks about the past | `memory search "..."` | Retrieve relevant entities |
| User says "remember this" | `memory store --type ...` | Direct store, not event |
| Debugging what the agent knows | `memory list`, `memory get` | Inspect state |

**Don't** use memory for: secrets/credentials, temporary conversation state (that's the context window), or large file contents (store a reference, not the content).

---

## OpenClaw Skill

### SKILL.md

```yaml
---
name: entity-memory
description: >
  Entity-based persistent memory with semantic search. Store facts about people,
  projects, tools, decisions. Search by meaning. Sub-linear growth over time.
  Use when the agent needs to remember, recall, or log observations.
metadata: {"openclaw":{"emoji":"🧠","requires":{"bins":["memory","docker"]}}}
---
```

### Instructions (in SKILL.md body)

Tell the agent:
- Use `memory event "<observation>"` after conversations and task completions
- Use `memory search "<query>"` before spawning sub-agents or answering questions about the past
- Use `memory store --type <type> --id <id> --content "<fact>"` when you learn something clearly structured
- Never put API keys, tokens, or passwords in memory commands
- Prefer `event` for raw observations; let the extraction cron handle entity creation

### Agent integration

Users should add the following to their agent's AGENTS.md (or equivalent workspace file)
to teach the agent how to use entity memory. This is a suggested template — adapt to fit
your agent's persona and workflow:

```markdown
## Memory

You have persistent entity memory. It stores facts about people, projects, tools,
preferences, and decisions. Facts are updated in place, not appended — the system
stays lean over time.

After conversations or task completions, log what happened:
  memory event "<what happened>"

When you learn a clear, structured fact, store it directly:
  memory store --type <person|project|tool|preference|decision> --id <key> --content "<fact>"

Before answering questions about the past or gathering context, search:
  memory search "<query>"

Entity types: person, project, tool, preference, decision.
Never store secrets (API keys, tokens, passwords) in memory.
```

---

## Docker Setup

### docker-compose.yml

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: openclaw-memory
    restart: unless-stopped
    ports:
      - "127.0.0.1:6333:6333"    # REST API — localhost only
      - "127.0.0.1:6334:6334"    # gRPC — localhost only
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  qdrant_data:
```

Key decisions:
- Ports bound to 127.0.0.1 — not accessible from network, only from localhost
- API key auth enabled via env var
- Persistent volume for data survival across container restarts
- Healthcheck for systemd/monitoring integration

### First-run initialization

The CLI should auto-create collections on first use if they don't exist.

```python
def ensure_collections(client: QdrantClient):
    for name in ["entities", "events", "decisions"]:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            # Text index for hybrid keyword search
            client.create_payload_index(
                collection_name=name,
                field_name="search_text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                ),
            )
            # Filterable fields
            client.create_payload_index(name, "type", field_schema=PayloadSchemaType.KEYWORD)
            client.create_payload_index(name, "entity_id", field_schema=PayloadSchemaType.KEYWORD)
```

---

## Python Package Structure

```
entity-memory/
├── pyproject.toml               # package config, dependencies, CLI entry point
├── README.md                    # usage docs
├── docker-compose.yml           # Qdrant container
├── skill/                       # OpenClaw skill (publishable to ClawHub)
│   └── SKILL.md
├── src/
│   └── entity_memory/
│       ├── __init__.py
│       ├── cli.py               # click CLI — entry point for all commands
│       ├── client.py            # Qdrant client wrapper (connect, ensure collections)
│       ├── embedder.py          # sentence-transformers wrapper, embed/embed_batch
│       ├── models.py            # dataclasses: Fact, Entity, Event
│       ├── merge.py             # merge, compact, drop_expired, find_duplicate, build_search_text
│       ├── extract.py           # event → entity extraction logic
│       ├── search.py            # search with vector + text + filter fusion
│       └── export.py            # export to JSON/markdown, import from JSON
└── tests/
    ├── test_merge.py            # unit tests for merge, compact, dedup
    ├── test_extract.py          # unit tests for extraction
    ├── test_search.py           # integration tests (needs Qdrant running)
    └── conftest.py              # fixtures: mock embedder, test Qdrant collection
```

### Dependencies

```toml
[project]
dependencies = [
    "qdrant-client>=1.12",
    "sentence-transformers>=3.0",
    "click>=8.0",
    "numpy>=1.24",
]

[project.scripts]
memory = "entity_memory.cli:main"
```

### Embedding model

Use `all-MiniLM-L6-v2`:
- 384 dimensions
- ~80MB on disk
- Fast: <10ms per embedding on CPU
- Good quality for short-text similarity (facts are short)
- The model downloads automatically on first use via sentence-transformers

---

## Cron Integration

### Heartbeat cron (every 55 min) — add extraction step

```json5
{
  id: "heartbeat-check",
  schedule: "*/55 * * * *",
  agentId: "main",   // replace with your agent id
  model: "openrouter/google/gemini-3.1-flash-lite",
  task: "Read HEARTBEAT.md. Run: memory extract --since 55m. Check tasks.json for 'todo' tasks — spawn agents. If nothing needs attention, reply HEARTBEAT_OK.",
}
```

### Weekly compaction + expiry (Sunday 2 AM)

```json5
{
  id: "memory-maintenance",
  schedule: "0 2 * * 0",
  agentId: "main",   // replace with your agent id
  model: "openrouter/google/gemini-3.1-flash-lite",
  task: "Run: memory compact. Run: memory expire. Run: memory export --format json > ~/memory-backup-$(date +%Y%m%d).json. Reply with counts of compacted entities and expired events.",
}
```

---

## Configuration

The CLI reads config from `~/.openclaw/memory.json` (or env vars):

```json
{
  "qdrant": {
    "url": "http://127.0.0.1:6333",
    "api_key_env": "QDRANT_API_KEY"
  },
  "embedder": {
    "model": "all-MiniLM-L6-v2",
    "device": "cpu"
  },
  "merge": {
    "dupe_threshold": 0.9,
    "max_facts": 20,
    "half_life_days": 30
  },
  "events": {
    "ttl_days": 30
  }
}
```

---

## Testing Strategy

### Unit tests (no Qdrant needed)

- `test_merge.py`:
  - Merge new facts into empty entity → all appended
  - Merge duplicate fact (cosine > 0.9) → hit_count incremented, not appended
  - Merge longer duplicate → text replaced, embedding updated
  - Expired facts dropped before merge
  - Mixed: some dupes, some new, some expired → correct outcome

- `test_compact.py`:
  - Entity with <= MAX_FACTS → unchanged
  - Entity with > MAX_FACTS → trimmed to MAX_FACTS
  - High hit_count facts survive over low hit_count
  - Recent facts survive over old facts
  - Permanent facts get boost over expiring facts

- `test_search_text.py`:
  - build_search_text produces expected format
  - Top facts (by score) appear first
  - Max 10 facts in search text

- `test_extract.py`:
  - Event text split into sentences
  - Sentence matching existing entity → merged
  - Sentence not matching → logged as unmatched
  - Event marked extracted after processing

### Integration tests (Qdrant running in Docker)

- `test_search.py`:
  - Store 3 entities, search by meaning → correct ranking
  - Search with type filter → only matching type returned
  - Search with keyword in text index → found even if semantic miss
  - Store entity, update with new facts, search → latest version returned

Use `conftest.py` to start a Qdrant container for tests (via `testcontainers` or a fixture that checks `docker ps`). Tests create a temporary collection and clean up after.

### Mock embedder for unit tests

```python
class MockEmbedder:
    """Deterministic embeddings for testing. Same text → same vector."""
    def embed(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[:48]]  # 48 dims for test speed
        norm = sum(v**2 for v in vec) ** 0.5
        return [v / (norm or 1.0) for v in vec]
```

Note: MockEmbedder won't produce meaningful cosine similarities (semantically
similar text won't score high). That's fine for testing merge mechanics. Use
the real model for integration tests that verify search quality.

---

## ClawHub Publication

### Skill structure for ClawHub

```
entity-memory/
├── SKILL.md                     # frontmatter + instructions
├── references/
│   └── entity-memory-guide.md   # detailed usage guide
├── scripts/
│   └── install.sh               # pip install + docker pull
└── README.md
```

### install.sh

```bash
#!/bin/bash
pip install entity-memory --break-system-packages
docker pull qdrant/qdrant:latest
# Start Qdrant if not running
if ! docker ps | grep -q openclaw-memory; then
    docker compose -f ~/.openclaw/skills/entity-memory/docker-compose.yml up -d
fi
```

---

## Security Considerations

- Qdrant ports bound to 127.0.0.1 only (not reachable from network)
- API key auth on Qdrant (loaded from env var, never in config files)
- QDRANT_API_KEY added to .env alongside other secrets
- Memory CLI never logs fact content at DEBUG level
- Export files should be treated as sensitive (contain all agent knowledge)
- The `memory` command runs via exec — ensure OpenClaw's exec tool is enabled

---

## What This Spec Does NOT Cover (out of scope)

- Multi-agent memory isolation (all agents share one Qdrant instance; filter by agent if needed later)
- Graph relationships between entities (payload filtering covers 90% of this; add a graph layer later if needed)
- Streaming/real-time memory updates (batch via cron is sufficient)
- Sparse vectors / SPLADE (start with dense + text index; upgrade if keyword recall is poor)
- Memory permissions / access control (single-user system, not multi-tenant)
