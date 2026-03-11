---
name: entity-memory
description: >
  Entity-based persistent memory with semantic search. Store facts about people,
  projects, tools, decisions. Search by meaning. Sub-linear growth over time.
  Use when the agent needs to remember, recall, or log observations.
metadata: {"openclaw":{"emoji":"🧠","requires":{"bins":["memory","docker"]}}}
---

# Entity Memory

You have persistent entity memory backed by Qdrant and local embeddings.
It stores facts about people, projects, tools, preferences, and decisions.
Facts are updated in place (upserted), not appended — the system stays lean over time.

## When to use

**After conversations or task completions**, log what happened:
```
memory event "<what happened>"
```

**When you learn a clear, structured fact**, store it directly:
```
memory store --type <person|project|tool|preference|decision> --id <key> --content "<fact>"
```

**Before answering questions about the past** or gathering context:
```
memory search "<query>"
```

## Entity types

- `person` — people and their roles, preferences, contacts
- `project` — codebases, services, architecture details
- `tool` — tools, libraries, configurations
- `preference` — user/team preferences and conventions
- `decision` — architectural and strategic choices (permanent, never expire)

## Rules

- Never store secrets (API keys, tokens, passwords) in memory
- Prefer `event` for raw observations; let extraction handle entity creation
- Use `store` when you're confident about the entity type and key
- Use `search` before spawning sub-agents to provide relevant context

## All commands

| Command | Purpose |
|---------|---------|
| `memory event "<text>"` | Log raw observation |
| `memory store --type T --id K --content "..."` | Upsert entity with fact |
| `memory search "<query>" [--type T] [--limit N]` | Semantic search |
| `memory extract [--since 55m \| --all]` | Process events into entities |
| `memory get <entity_id>` | Show entity detail |
| `memory list [--type T]` | List all entities |
| `memory delete <entity_id>` | Delete entity |
| `memory compact [--max-facts N]` | Prune overgrown entities |
| `memory expire` | GC expired events and facts |
| `memory export [--format json\|md]` | Backup entities |
| `memory import <file>` | Restore from backup |
| `memory stats` | Collection sizes |
| `memory init` | Create collections |
