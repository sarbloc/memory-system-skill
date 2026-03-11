"""Merge, compact, expiry, and search text generation — all deterministic, no LLM."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

import numpy as np

from entity_memory.models import Entity, Fact


# ── Constants ────────────────────────────────────────────

DUPE_THRESHOLD = 0.9
MAX_FACTS = 20
HALF_LIFE_DAYS = 30


# ── Embedder protocol (satisfied by both Embedder and MockEmbedder) ──

class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


# ── Helpers ──────────────────────────────────────────────

def ensure_embedded(fact: Fact, embedder: EmbedderLike) -> None:
    """Lazily embed a fact if it doesn't already have an embedding."""
    if fact.embedding is None:
        fact.embedding = embedder.embed(fact.text)


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_, b_ = np.array(a), np.array(b)
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return 0.0 if denom == 0 else float(np.dot(a_, b_) / denom)


def find_duplicate(
    new: Fact, existing: list[Fact], embedder: EmbedderLike
) -> Optional[Fact]:
    """Find the existing fact most similar to `new`. Returns it if above threshold, else None."""
    ensure_embedded(new, embedder)
    best_match = None
    best_score = 0.0
    for ex in existing:
        ensure_embedded(ex, embedder)
        score = cosine_sim(new.embedding, ex.embedding)
        if score > best_score:
            best_score = score
            best_match = ex
    if best_score >= DUPE_THRESHOLD:
        return best_match
    return None


def drop_expired(facts: list[Fact], now: datetime) -> list[Fact]:
    """Remove facts whose expiry date has passed."""
    return [
        f for f in facts
        if f.expires is None or datetime.fromisoformat(f.expires) > now
    ]


# ── Fact scoring (shared by compact and build_search_text) ──

def fact_score(fact: Fact, now: datetime) -> float:
    """Score a fact by frequency × recency × permanence."""
    last = datetime.fromisoformat(fact.last_seen or fact.added)
    age_days = (now - last).total_seconds() / 86400

    frequency = 1 + float(np.log1p(fact.hit_count))
    recency = 2 ** (-age_days / HALF_LIFE_DAYS)
    permanence = 1.2 if fact.expires is None else 1.0

    return frequency * recency * permanence


# ── Core operations ──────────────────────────────────────

def merge(
    entity: Entity,
    new_facts: list[Fact],
    embedder: EmbedderLike,
    now: datetime | None = None,
) -> Entity:
    """Merge new facts into an entity.

    For each new fact:
      - If a semantic duplicate exists (cosine >= 0.9):
          update last_seen and hit_count; replace text if new is longer.
      - Otherwise: append as a new fact.

    Expired facts are dropped first.
    """
    now = now or datetime.utcnow()
    today = now.date().isoformat()

    entity.facts = drop_expired(entity.facts, now)

    for new_fact in new_facts:
        ensure_embedded(new_fact, embedder)
        dupe = find_duplicate(new_fact, entity.facts, embedder)

        if dupe is not None:
            dupe.hit_count += 1
            dupe.last_seen = today
            if len(new_fact.text) > len(dupe.text):
                dupe.text = new_fact.text
                dupe.embedding = new_fact.embedding
        else:
            new_fact.last_seen = today
            entity.facts.append(new_fact)

    entity.last_updated = now.isoformat()
    return entity


def compact(entity: Entity, max_facts: int = MAX_FACTS, now: datetime | None = None) -> Entity:
    """Trim facts to max_facts by keeping the highest-scored ones.

    Score = frequency × recency × permanence (see fact_score).
    """
    if len(entity.facts) <= max_facts:
        return entity

    now = now or datetime.utcnow()
    scored = sorted(entity.facts, key=lambda f: fact_score(f, now), reverse=True)
    entity.facts = scored[:max_facts]
    entity.last_updated = now.isoformat()
    return entity


def build_search_text(entity: Entity, now: datetime | None = None) -> str:
    """Build the search text string used for the entity's dense vector in Qdrant.

    Format: "[{type}] {entity_id}. {fact_1}. {fact_2}. ... {fact_10}"
    Facts sorted by score, top 10.
    """
    now = now or datetime.utcnow()
    parts = [f"[{entity.type}] {entity.id}"]
    sorted_facts = sorted(entity.facts, key=lambda f: fact_score(f, now), reverse=True)
    for f in sorted_facts[:10]:
        parts.append(f.text)
    return ". ".join(parts)
