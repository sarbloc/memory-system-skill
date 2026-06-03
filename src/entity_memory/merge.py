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
        # Dedup only against facts valid *today* (issue #21). Superseded/expired
        # history is immutable: a re-asserted fact must not revive or mutate it —
        # it appends as a new current fact instead, leaving the historical record
        # intact for as-of queries. Using valid_at (not is_current) keeps a fact
        # whose supersession is future-effective as a dedup target until its date.
        live = [f for f in entity.facts if f.valid_at(today)]
        dupe = find_duplicate(new_fact, live, embedder)

        if dupe is not None:
            dupe.hit_count += 1
            dupe.last_seen = today
            # Keep the earliest known valid-time start. A re-assertion that
            # carries an earlier valid_from means we learned the fact was true
            # sooner than recorded; widening the window backwards must not be
            # lost just because the texts deduped (issue #21).
            if new_fact.valid_from is not None:
                existing_start = dupe.valid_from or dupe.added
                if new_fact.valid_from < existing_start:
                    dupe.valid_from = new_fact.valid_from
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

    Current facts always outrank superseded ones, so historical facts are the
    first to drop when over the limit (issue #21); within each group the order
    is by score (frequency × recency × permanence — see fact_score). Superseded
    facts are only retained if there is room left after every current fact,
    preserving recent history for as-of queries without crowding out live facts.
    """
    if len(entity.facts) <= max_facts:
        return entity

    now = now or datetime.utcnow()
    scored = sorted(
        entity.facts,
        key=lambda f: (f.is_current, fact_score(f, now)),
        reverse=True,
    )
    entity.facts = scored[:max_facts]
    entity.last_updated = now.isoformat()
    return entity


def build_search_text(entity: Entity, now: datetime | None = None) -> str:
    """Build the search text string used for the entity's dense vector in Qdrant.

    Format: "[{type}] {entity_id}. {fact_1}. {fact_2}. ... {fact_10}"
    Current facts only, sorted by score, top 10. Superseded facts are excluded
    (issue #21) so stale state stops feeding the entity's search vector.
    """
    now = now or datetime.utcnow()
    today = now.date().isoformat()
    parts = [f"[{entity.type}] {entity.id}"]
    # Facts valid as of today drive the vector. Using valid_at (not is_current)
    # means a fact superseded with a *future* effective date keeps feeding the
    # index until that date, instead of dropping out the moment supersession is
    # recorded (issue #21, Codex review of PR #23).
    valid = [f for f in entity.facts if f.valid_at(today)]
    sorted_facts = sorted(valid, key=lambda f: fact_score(f, now), reverse=True)
    for f in sorted_facts[:10]:
        parts.append(f.text)
    return ". ".join(parts)


def mark_superseded(
    entity: Entity, target: str, *, by: str, on_date: str
) -> Optional[Fact]:
    """Mark the current fact identified by ``target`` as superseded.

    ``target`` matches a current fact by exact text or by source id. The first
    matching current fact gets ``superseded_at = on_date`` and
    ``superseded_by = by`` (the replacing fact's text). Already-superseded facts
    are skipped. Returns the fact that was superseded, or None if nothing
    matched.

    Deterministic by design (issue #21): this records a supersession the caller
    has already decided on. The core never *infers* that one fact supersedes
    another — that semantic judgement is the LLM agent's job, mirroring how
    extract.py leaves entity creation to the agent.
    """
    for f in entity.facts:
        if not f.is_current:
            continue
        if f.text == target or f.source == target:
            f.superseded_at = on_date
            f.superseded_by = by
            return f
    return None
