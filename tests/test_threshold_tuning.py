"""Real-embedder guard for the tuned MATCH_THRESHOLD (issue #17).

Unlike the rest of the suite (which uses the non-semantic MockEmbedder, where
cosines are meaningless), this loads the real all-MiniLM-L6-v2 model and asserts
the hand-labelled fixture in ``tests/data/match_pairs.json`` classifies as
intended at the *shipped* ``MATCH_THRESHOLD`` — driven through the real matcher
(``build_entity_index`` + ``EntityIndex.best_match``), not a parallel
reimplementation.

It is the regression lock for the tuning: push the threshold back toward 0.7,
or regress per-fact matching, and the labelled positives stop matching here.

Skipped automatically when sentence-transformers or the cached model isn't
available (e.g. a bare CI box), so it never blocks the mock-only fast path.
"""

import json
from pathlib import Path

import pytest

from entity_memory.extract import MATCH_THRESHOLD, build_entity_index
from entity_memory.models import Entity, Fact

FIXTURE = Path(__file__).parent / "data" / "match_pairs.json"


def _load_pairs() -> list[dict]:
    return json.loads(FIXTURE.read_text())["pairs"]


def _entity_of(pair: dict) -> Entity:
    spec = pair["entity"]
    facts = [Fact(text=t, added="2026-01-01", source="fixture") for t in spec["facts"]]
    return Entity(
        id=spec["id"], type=spec["type"], facts=facts,
        last_updated="2026-01-01T00:00:00",
    )


@pytest.fixture(scope="module")
def real_embedder():
    """The real MiniLM model, or skip the whole module if it can't load."""
    try:
        from entity_memory.embedder import Embedder

        return Embedder()
    except Exception as exc:  # ImportError, blocked download, missing cache…
        pytest.skip(f"real all-MiniLM-L6-v2 unavailable: {exc}")


@pytest.mark.parametrize(
    "pair",
    _load_pairs(),
    ids=lambda p: f"{p['entity']['id']}|{p['sentence'][:30]}",
)
def test_labelled_pair_classifies_at_shipped_threshold(real_embedder, pair):
    """Each labelled pair matches (or not) as intended at MATCH_THRESHOLD.

    Hard positives (semantically valid but scoring below the separable band,
    flagged ``hard`` in the fixture) are not enforced — no global gate catches
    them without also admitting a negative.
    """
    index = build_entity_index([_entity_of(pair)], real_embedder)
    sent_vec = real_embedder.embed(pair["sentence"])
    matched = index.best_match(sent_vec, MATCH_THRESHOLD) is not None

    if pair["should_match"]:
        if pair.get("hard"):
            pytest.skip("hard positive: unseparable by a global gate, not enforced")
        assert matched, (
            f"expected enrichment at threshold {MATCH_THRESHOLD}: "
            f"{pair['sentence']!r} should match {pair['entity']['id']}"
        )
    else:
        assert not matched, (
            f"false positive at threshold {MATCH_THRESHOLD}: "
            f"{pair['sentence']!r} wrongly enriched {pair['entity']['id']}"
        )
