"""Event → entity extraction logic.

Splits event text into sentences, matches each against existing entities.
Matched sentences are merged as new facts (enrichment of existing entities).
Unmatched sentences are returned as candidates but NOT auto-created as
entities here — entity *creation* is deliberately left to an external LLM
agent (via the ``memory_store`` tool). This module stays dependency-light
(local embedder only) and does mechanical cosine matching for enrichment.

``ExtractionResult.matched_event_ids`` records which events had at least one
sentence match an existing entity, so callers can mark *only* those events
extracted and leave fully-unmatched events as new-entity candidates rather
than silently burning them.

Performance: the entity corpus is embedded **once** per run via
``build_entity_index`` and reused across every sentence. The previous code
re-embedded every entity inside the per-sentence loop, giving
``O(events × sentences × entities)`` embedding calls — which pegged all cores
for hours on bulk imports (see issue #12). Matching against a prebuilt index is
plain cosine arithmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from entity_memory.merge import build_search_text
from entity_memory.models import Entity


MATCH_THRESHOLD = 0.7

# A prebuilt entity index: each entity paired with its search_text embedding,
# computed once so the per-sentence match loop is embedding-free.
EntityIndex = list["tuple[Entity, list[float]]"]


class EmbedderLike(Protocol):
    def embed(self, text: str) -> list[float]: ...


@dataclass
class ExtractionResult:
    matched: list[tuple[str, Entity]]  # (sentence, matched entity)
    unmatched: list[str]  # sentences with no entity match
    events_processed: int
    matched_event_ids: set[str]  # ids of events with >=1 sentence matched to an entity


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


def build_entity_index(entities: list[Entity], embedder: EmbedderLike) -> EntityIndex:
    """Embed each entity's search_text once, up front.

    Returns ``[(entity, vector), ...]``. Building this outside the per-sentence
    loop is what removes the ``O(events × sentences × entities)`` embedding
    blowup (issue #12): a run embeds the corpus once, then matches with cosine
    arithmetic only.
    """
    return [(entity, embedder.embed(build_search_text(entity))) for entity in entities]


def match_sentence_to_index(
    sent_vec: list[float],
    index: EntityIndex,
    threshold: float = MATCH_THRESHOLD,
) -> Entity | None:
    """Best-matching entity for a precomputed sentence vector against an index.

    Pure cosine arithmetic — no embedding. Returns the entity if the best
    similarity >= threshold, else None.
    """
    best_entity = None
    best_score = 0.0
    for entity, entity_vec in index:
        score = cosine_sim(sent_vec, entity_vec)
        if score > best_score:
            best_score = score
            best_entity = entity
    return best_entity if best_score >= threshold else None


def match_sentence_to_entity(
    sentence: str,
    entities: list[Entity],
    embedder: EmbedderLike,
    threshold: float = MATCH_THRESHOLD,
) -> Entity | None:
    """Find the best matching entity for a sentence by cosine similarity.

    Convenience wrapper that builds a one-shot index. For bulk work, build the
    index once with ``build_entity_index`` and call ``match_sentence_to_index``
    so the corpus is not re-embedded per sentence.
    """
    if not entities:
        return None
    index = build_entity_index(entities, embedder)
    return match_sentence_to_index(embedder.embed(sentence), index, threshold)


def extract_events_with_index(
    events: list[dict],
    index: EntityIndex,
    embedder: EmbedderLike,
    now: datetime | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> ExtractionResult:
    """Match sentences in ``events`` against a prebuilt entity index.

    Each sentence is embedded once; matching is cosine-only against the index.
    """
    matched: list[tuple[str, Entity]] = []
    unmatched: list[str] = []
    matched_event_ids: set[str] = set()

    for event in events:
        for sentence in split_sentences(event["text"]):
            entity = match_sentence_to_index(embedder.embed(sentence), index, threshold)
            if entity is not None:
                matched.append((sentence, entity))
                matched_event_ids.add(event["id"])
            else:
                unmatched.append(sentence)

    return ExtractionResult(
        matched=matched,
        unmatched=unmatched,
        events_processed=len(events),
        matched_event_ids=matched_event_ids,
    )


def extract_events(
    events: list[dict],
    entities: list[Entity],
    embedder: EmbedderLike,
    now: datetime | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> ExtractionResult:
    """Process a list of events, matching sentences to existing entities.

    Builds the entity index once (``build_entity_index``) then matches, so
    embedding calls scale as ``len(entities) + total_sentences`` rather than
    their product.

    Args:
        events: list of event dicts with at least "text" and "id" keys
        entities: all existing entities to match against
        embedder: embedding model
        now: current time (accepted for caller compatibility; matching itself
            is time-independent)
        threshold: cosine gate for a sentence→entity match

    Returns:
        ExtractionResult with matched/unmatched sentences
    """
    index = build_entity_index(entities, embedder)
    return extract_events_with_index(
        events, index, embedder, now=now, threshold=threshold
    )
