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

Matching shape (issue #17): a sentence is compared against each of an entity's
facts *individually* and scored best-of, NOT against the concatenated
``build_search_text`` blob. The blob is the centroid of up to 10 different
facts plus a ``[type] id`` prefix; averaging them together pulls cosine down so
far that even on-topic sentences rarely clear the gate, and enrichment almost
never fires (facts fragment into new entities instead). Per-fact matching
compares like with like. ``build_search_text`` is unchanged — storage and
search keep the blob; only this enrich path moved to per-fact.

Performance: every fact is embedded **once** per run via
``build_entity_index`` and the corpus is packed into one normalized matrix, so
each sentence match is a single ``matrix @ vec`` dot-product (issue #12). The
previous code re-embedded every entity inside the per-sentence loop, giving
``O(events × sentences × entities)`` embedding calls. Per-fact indexing adds
~10× more rows than per-entity, but a matmul over a few thousand rows is
microseconds — the cost stays in arithmetic, not embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np

from entity_memory.models import Entity


# Cosine gate for a sentence→fact match. Tuned (issue #17) against the
# hand-labelled fixture in tests/data/match_pairs.json with the real
# all-MiniLM-L6-v2 model: per-fact cosines put clearly-related pairs at
# 0.34–0.73 and unrelated/trap pairs at <=0.16, leaving a clean 0.16→0.34 gap.
# 0.32 sits in that gap — precision 1.0 with a ~0.16 margin above the worst
# negative, while still catching weak-but-genuine positives. The old 0.7 (vs a
# diluted entity-blob vector) almost never fired, fragmenting facts into new
# entities. Re-run scripts/tune_match_threshold.py to revisit.
MATCH_THRESHOLD = 0.32


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
    a_, b_ = np.array(a), np.array(b)
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return 0.0 if denom == 0 else float(np.dot(a_, b_) / denom)


@dataclass
class EntityIndex:
    """Prebuilt per-fact match index over an entity corpus.

    Every fact of every entity is embedded once and stored as one row of a
    normalized matrix, so a sentence match is a single ``matrix @ sent_vec``
    dot-product followed by an argmax. ``row_owner[r]`` maps matrix row ``r``
    back to ``entities[row_owner[r]]`` — the entity that fact belongs to.

    Matching best-of over an entity's facts then taking the best entity reduces
    to a global argmax over all rows, since ``max`` of per-entity ``max`` is the
    global ``max``. No per-entity grouping needed.
    """

    entities: list[Entity]
    matrix: np.ndarray  # (n_rows, dim), L2-normalized rows
    row_owner: np.ndarray  # (n_rows,) int → index into `entities`

    def best_match(
        self, sent_vec: list[float], threshold: float = MATCH_THRESHOLD
    ) -> Entity | None:
        """Entity owning the fact most similar to ``sent_vec``, or None.

        Pure arithmetic — no embedding. Returns the entity whose best-matching
        fact has cosine >= threshold, else None.
        """
        if self.matrix.shape[0] == 0:
            return None
        s = np.asarray(sent_vec, dtype=np.float32)
        norm = float(np.linalg.norm(s))
        if norm == 0.0:
            return None
        scores = self.matrix @ (s / norm)  # cosine: rows already normalized
        best = int(np.argmax(scores))
        if float(scores[best]) < threshold:
            return None
        return self.entities[int(self.row_owner[best])]


def build_entity_index(entities: list[Entity], embedder: EmbedderLike) -> EntityIndex:
    """Embed every fact once, up front, into a normalized match matrix.

    One row per fact (the ``[type] id`` header is deliberately excluded — entity
    ids are not natural language and only add noise; see issue #17). Building
    this outside the per-sentence loop is what removes the
    ``O(events × sentences × entities)`` embedding blowup (issue #12): a run
    embeds the corpus once, then matches with matrix arithmetic only.

    An entity with no facts contributes no rows and so can never be enriched —
    correct: there is nothing to match a sentence against.
    """
    ents = list(entities)
    rows: list[list[float]] = []
    row_owner: list[int] = []
    for i, entity in enumerate(ents):
        for fact in entity.facts:
            rows.append(embedder.embed(fact.text))
            row_owner.append(i)

    if rows:
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)

    return EntityIndex(
        entities=ents,
        matrix=matrix,
        row_owner=np.asarray(row_owner, dtype=np.int64),
    )


def match_sentence_to_index(
    sent_vec: list[float],
    index: EntityIndex,
    threshold: float = MATCH_THRESHOLD,
) -> Entity | None:
    """Best-matching entity for a precomputed sentence vector against an index.

    Thin wrapper over ``EntityIndex.best_match`` kept for call-site stability.
    """
    return index.best_match(sent_vec, threshold)


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
    embedding calls scale as ``total_facts + total_sentences`` rather than
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
