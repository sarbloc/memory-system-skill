"""
Entity merge logic — pure code, no LLM.
Dedup, merge, expire, compact all handled deterministically.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


# ── Data structures ──────────────────────────────────────

@dataclass
class Fact:
    text: str
    added: str                               # ISO date
    source: str                              # event ID that produced this
    expires: Optional[str] = None            # ISO date or None (permanent)
    last_seen: Optional[str] = None          # updated on duplicate hit
    hit_count: int = 1                       # how many times this fact was reinforced
    embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class Entity:
    id: str                                  # "person:alice", "project:dashboard"
    type: str                                # person | project | tool | preference | decision
    facts: list[Fact] = field(default_factory=list)
    last_updated: str = ""


# ── Embedding ────────────────────────────────────────────

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_, b_ = np.array(a), np.array(b)
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return 0.0 if denom == 0 else float(np.dot(a_, b_) / denom)


# ── Merge logic ──────────────────────────────────────────

DUPE_THRESHOLD = 0.9
MAX_FACTS = 20


def ensure_embedded(fact: Fact, embedder: Embedder) -> None:
    if fact.embedding is None:
        fact.embedding = embedder.embed(fact.text)


def find_duplicate(new: Fact, existing: list[Fact], embedder: Embedder) -> Optional[Fact]:
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
    return [
        f for f in facts
        if f.expires is None or datetime.fromisoformat(f.expires) > now
    ]


def merge(entity: Entity, new_facts: list[Fact], embedder: Embedder, now: datetime | None = None) -> Entity:
    """
    Merge new facts into an entity. No LLM.

    For each new fact:
      - If a semantic duplicate exists (cosine >= 0.9):
          update its last_seen and hit_count (reinforcement)
          if the new text is longer, replace the old text (richer version wins)
      - If no duplicate: append as new fact

    Then drop expired facts.
    """
    now = now or datetime.utcnow()
    today = now.date().isoformat()

    # Expire stale facts first
    entity.facts = drop_expired(entity.facts, now)

    for new_fact in new_facts:
        ensure_embedded(new_fact, embedder)
        dupe = find_duplicate(new_fact, entity.facts, embedder)

        if dupe is not None:
            # Reinforce: bump count and timestamp
            dupe.hit_count += 1
            dupe.last_seen = today
            # If new version is longer (richer), replace text and re-embed
            if len(new_fact.text) > len(dupe.text):
                dupe.text = new_fact.text
                dupe.embedding = new_fact.embedding
        else:
            # Genuinely new fact
            new_fact.last_seen = today
            entity.facts.append(new_fact)

    entity.last_updated = now.isoformat()
    return entity


def compact(entity: Entity) -> Entity:
    """
    Trim facts to MAX_FACTS without an LLM.

    Strategy: score each fact, keep the top MAX_FACTS.
    Score = hit_count * recency_weight
    This keeps frequently-reinforced and recent facts, drops one-off old ones.
    """
    if len(entity.facts) <= MAX_FACTS:
        return entity

    now = datetime.utcnow()

    def score(f: Fact) -> float:
        # Recency: days since last seen, half-life of 30 days
        last = datetime.fromisoformat(f.last_seen) if f.last_seen else datetime.fromisoformat(f.added)
        age_days = (now - last).total_seconds() / 86400
        recency = 2 ** (-age_days / 30)  # half-life decay

        # Frequency: log scale so one fact mentioned 100x doesn't dominate everything
        frequency = 1 + np.log1p(f.hit_count)

        # Permanent facts (no expiry) get a small boost
        permanence = 1.2 if f.expires is None else 1.0

        return float(frequency * recency * permanence)

    scored = sorted(entity.facts, key=score, reverse=True)
    entity.facts = scored[:MAX_FACTS]
    entity.last_updated = now.isoformat()
    return entity


# ── The summary vector for Qdrant ────────────────────────

def build_search_text(entity: Entity) -> str:
    """
    Combine entity metadata + top facts into a single string for embedding.
    This becomes the dense vector stored in Qdrant.
    No LLM — just string concatenation with structure.
    """
    parts = [f"[{entity.type}] {entity.id}"]

    # Sort facts by score (most important first) so the embedding
    # is weighted toward the most relevant information
    now = datetime.utcnow()
    def recency(f: Fact) -> float:
        last = datetime.fromisoformat(f.last_seen or f.added)
        age = (now - last).total_seconds() / 86400
        return f.hit_count * (2 ** (-age / 30))

    sorted_facts = sorted(entity.facts, key=recency, reverse=True)

    # Take top 10 facts for the search text (keeps embedding focused)
    for f in sorted_facts[:10]:
        parts.append(f.text)

    return ". ".join(parts)


def embed_entity(entity: Entity, embedder: Embedder) -> list[float]:
    """Generate the dense vector for Qdrant storage."""
    search_text = build_search_text(entity)
    return embedder.embed(search_text)


# ── Demo ─────────────────────────────────────────────────

if __name__ == "__main__":

    # Use a mock embedder for demo (real one needs sentence-transformers)
    class MockEmbedder:
        """Hashes text into a fake 8-dim vector for demonstration."""
        def embed(self, text: str) -> list[float]:
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:8]]
            norm = sum(v**2 for v in vec) ** 0.5
            return [v / norm for v in vec]

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    embedder = MockEmbedder()
    today = datetime(2026, 3, 10)
    yesterday = today - timedelta(days=1)

    # ── Create entity with initial facts ─────────────────

    alice = Entity(id="person:alice", type="person")

    initial_facts = [
        Fact(text="Manages the auth team", added="2026-02-15", source="event:001"),
        Fact(text="Prefers Slack for async communication", added="2026-02-20", source="event:002"),
        Fact(
            text="On vacation until March 15",
            added="2026-03-08",
            source="event:003",
            expires="2026-03-16",
        ),
    ]

    alice = merge(alice, initial_facts, embedder, now=yesterday)
    print(f"After initial merge: {len(alice.facts)} facts")
    for f in alice.facts:
        print(f"  [{f.hit_count}x] {f.text}")

    # ── Merge new facts (some duplicates, some new) ──────

    new_facts = [
        # Duplicate: "manages auth" said differently
        Fact(text="Alice leads the authentication team", added="2026-03-10", source="event:010"),
        # Genuinely new
        Fact(text="Approved the auth API pull request", added="2026-03-10", source="event:011"),
        # Richer version of existing fact
        Fact(
            text="Prefers Slack for async communication, especially the #auth-team channel",
            added="2026-03-10",
            source="event:012",
        ),
    ]

    alice = merge(alice, new_facts, embedder, now=today)
    print(f"\nAfter second merge: {len(alice.facts)} facts")
    for f in alice.facts:
        print(f"  [{f.hit_count}x] {f.text}")

    # With real embeddings:
    #   "Manages the auth team" vs "Alice leads the authentication team"
    #   cosine > 0.9 → duplicate → hit_count bumps to 2, text stays (new isn't longer)
    #
    #   "Prefers Slack..." vs "Prefers Slack... especially #auth-team channel"
    #   cosine > 0.9 → duplicate → but new is longer → text replaced, hit_count = 2
    #
    #   "Approved the auth API pull request"
    #   no match → appended as new fact

    # ── Expiry ───────────────────────────────────────────

    march_20 = datetime(2026, 3, 20)
    alice.facts = drop_expired(alice.facts, march_20)
    print(f"\nAfter expiry (March 20): {len(alice.facts)} facts")
    for f in alice.facts:
        print(f"  [{f.hit_count}x] {f.text} {'[TEMP]' if f.expires else ''}")

    # "On vacation until March 15" → expired → gone

    # ── Compaction ───────────────────────────────────────

    # Simulate an entity that accumulated too many facts
    bloated = Entity(id="project:dashboard", type="project")
    bloated.facts = [
        Fact(
            text=f"Sprint {i} completed task #{i}",
            added=(today - timedelta(days=60 - i)).date().isoformat(),
            source=f"event:{i:03d}",
            hit_count=(1 if i < 15 else 3),  # recent tasks referenced more
            last_seen=(today - timedelta(days=60 - i)).date().isoformat(),
        )
        for i in range(25)
    ]

    print(f"\nBefore compaction: {len(bloated.facts)} facts")
    bloated = compact(bloated)
    print(f"After compaction:  {len(bloated.facts)} facts (kept top {MAX_FACTS})")
    print("Kept facts (most relevant by frequency × recency):")
    for f in bloated.facts[:5]:
        print(f"  [{f.hit_count}x] {f.text}")

    # ── Search text generation ───────────────────────────

    print(f"\nSearch text for Qdrant embedding:")
    print(f"  {build_search_text(alice)}")
