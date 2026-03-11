"""Embedding wrapper: real model and mock for testing."""

from __future__ import annotations

import hashlib

import numpy as np


class Embedder:
    """Wraps sentence-transformers for local embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class MockEmbedder:
    """Deterministic hash-based embeddings for testing.

    Same text always produces the same vector. Vectors are normalized.
    Note: cosine similarities won't be semantically meaningful.
    """

    def __init__(self, dims: int = 48):
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        raw = [float(b) / 255.0 for b in h[: self.dims]]
        norm = sum(v**2 for v in raw) ** 0.5
        return [v / (norm or 1.0) for v in raw]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_, b_ = np.array(a), np.array(b)
    denom = np.linalg.norm(a_) * np.linalg.norm(b_)
    return 0.0 if denom == 0 else float(np.dot(a_, b_) / denom)
