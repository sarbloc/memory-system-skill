#!/usr/bin/env python
"""Tune the extract MATCH_THRESHOLD against the hand-labelled fixture (issue #17).

Runs the REAL all-MiniLM-L6-v2 model (the mock embedder is non-semantic and
useless for this) over ``tests/data/match_pairs.json``, scoring each sentence
against its entity's facts the same way the live matcher does: best-of cosine
across the entity's per-fact vectors, header excluded. It then sweeps candidate
thresholds and reports precision / recall / F1 / accuracy so the gate is picked
from data, not vibes.

Run from the repo root:

    HF_HUB_OFFLINE=1 python scripts/tune_match_threshold.py

Read-only: it does not touch Qdrant or change any constant. Use the printed
recommendation to set ``MATCH_THRESHOLD`` in ``src/entity_memory/extract.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from entity_memory.embedder import Embedder, cosine_sim

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "data" / "match_pairs.json"
THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(21)]  # 0.300 … 0.800


def best_fact_cosine(embedder, sentence: str, facts: list[str]) -> float:
    """Best cosine of the sentence against any of the entity's facts."""
    s = embedder.embed(sentence)
    return max(cosine_sim(s, embedder.embed(f)) for f in facts)


def confusion(scored: list[tuple[bool, float]], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for should_match, score in scored:
        predicted = score >= threshold
        if should_match and predicted:
            tp += 1
        elif should_match and not predicted:
            fn += 1
        elif not should_match and predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(scored) if scored else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def main() -> None:
    data = json.loads(FIXTURE.read_text())
    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} labelled pairs from {FIXTURE.name}")
    print("Loading all-MiniLM-L6-v2 (real model)…")
    embedder = Embedder()

    scored: list[tuple[bool, float]] = []
    rows = []
    for p in pairs:
        score = best_fact_cosine(embedder, p["sentence"], p["entity"]["facts"])
        scored.append((p["should_match"], score))
        rows.append((p["should_match"], score, p["entity"]["id"], p["sentence"]))

    # Per-pair scores, sorted high→low: the gap between the lowest positive and
    # the highest negative is the room the threshold has to separate them.
    print("\n=== Per-pair best-fact cosine (sorted) ===")
    print(f"{'label':>9}  {'cosine':>6}  entity / sentence")
    for should_match, score, eid, sent in sorted(rows, key=lambda r: r[1], reverse=True):
        label = "MATCH" if should_match else "no"
        print(f"{label:>9}  {score:6.3f}  {eid} — {sent}")

    pos = [s for m, s in scored if m]
    neg = [s for m, s in scored if not m]
    print(f"\npositives: min={min(pos):.3f} max={max(pos):.3f}  ({len(pos)} pairs)")
    print(f"negatives: min={min(neg):.3f} max={max(neg):.3f}  ({len(neg)} pairs)")
    print(f"separation gap (lowest positive - highest negative): "
          f"{min(pos) - max(neg):+.3f}")

    # Threshold sweep.
    print("\n=== Threshold sweep ===")
    print(f"{'thresh':>6}  {'TP':>2} {'FP':>2} {'FN':>2} {'TN':>2}  "
          f"{'prec':>5} {'recall':>6} {'F1':>5} {'acc':>5}")
    best = None
    for t in THRESHOLDS:
        c = confusion(scored, t)
        marker = ""
        # Prefer max F1; break ties toward the higher threshold (more precision).
        if best is None or c["f1"] > best[1]["f1"] or (
            c["f1"] == best[1]["f1"] and t > best[0]
        ):
            best = (t, c)
        print(f"{t:6.3f}  {c['tp']:2d} {c['fp']:2d} {c['fn']:2d} {c['tn']:2d}  "
              f"{c['precision']:5.2f} {c['recall']:6.2f} {c['f1']:5.2f} {c['accuracy']:5.2f}{marker}")

    bt, bc = best
    print(f"\nRecommended threshold (max F1): {bt:.3f}")
    print(f"  precision={bc['precision']:.2f} recall={bc['recall']:.2f} "
          f"F1={bc['f1']:.2f} accuracy={bc['accuracy']:.2f}  "
          f"(TP={bc['tp']} FP={bc['fp']} FN={bc['fn']} TN={bc['tn']})")
    print("\nThis is a recommendation, not a commit. Eyeball the per-pair table "
          "and the sweep, then set MATCH_THRESHOLD deliberately.")


if __name__ == "__main__":
    main()
