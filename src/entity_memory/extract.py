"""Event → entity extraction logic.

Splits event text into sentences, matches each against existing entities.
Matched sentences are merged as new facts. Unmatched sentences are logged
as candidates but NOT auto-created as entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from entity_memory.models import Entity, Fact


MATCH_THRESHOLD = 0.7


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class ExtractionResult:
    matched: list[tuple[str, Entity]]  # (sentence, matched entity)
    unmatched: list[str]  # sentences with no entity match
    events_processed: int


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings and very short fragments
    return [p.strip() for p in parts if len(p.strip()) >= 5]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np

    a_, b_ = np.array(a), np.array(b)
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return 0.0 if denom == 0 else float(np.dot(a_, b_) / denom)


def match_sentence_to_entity(
    sentence: str,
    entities: list[Entity],
    embedder: EmbedderLike,
    threshold: float = MATCH_THRESHOLD,
) -> Entity | None:
    """Find the best matching entity for a sentence by cosine similarity.

    Compares the sentence embedding against each entity's search_text embedding.
    Returns the entity if similarity >= threshold, else None.
    """
    from entity_memory.merge import build_search_text

    if not entities:
        return None

    sent_vec = embedder.embed(sentence)
    best_entity = None
    best_score = 0.0

    for entity in entities:
        search_text = build_search_text(entity)
        entity_vec = embedder.embed(search_text)
        score = cosine_sim(sent_vec, entity_vec)
        if score > best_score:
            best_score = score
            best_entity = entity

    if best_score >= threshold:
        return best_entity
    return None


def extract_events(
    events: list[dict],
    entities: list[Entity],
    embedder: EmbedderLike,
    now: datetime | None = None,
) -> ExtractionResult:
    """Process a list of events, matching sentences to existing entities.

    Args:
        events: list of event dicts with at least "text" and "id" keys
        entities: all existing entities to match against
        embedder: embedding model
        now: current time for fact timestamps

    Returns:
        ExtractionResult with matched/unmatched sentences
    """
    now = now or datetime.utcnow()
    today = now.date().isoformat()

    matched = []
    unmatched = []

    for event in events:
        sentences = split_sentences(event["text"])
        for sentence in sentences:
            entity = match_sentence_to_entity(sentence, entities, embedder)
            if entity is not None:
                matched.append((sentence, entity))
            else:
                unmatched.append(sentence)

    return ExtractionResult(
        matched=matched,
        unmatched=unmatched,
        events_processed=len(events),
    )
