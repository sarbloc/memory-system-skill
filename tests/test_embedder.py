"""Tests for embedder and cosine similarity."""

import math

from entity_memory.embedder import MockEmbedder, cosine_sim


def test_mock_embedder_deterministic(embedder):
    a = embedder.embed("hello world")
    b = embedder.embed("hello world")
    assert a == b


def test_mock_embedder_normalized(embedder):
    vec = embedder.embed("test string")
    norm = math.sqrt(sum(v**2 for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_mock_embedder_different_texts(embedder):
    a = embedder.embed("hello")
    b = embedder.embed("goodbye")
    assert a != b


def test_mock_embedder_dims():
    e = MockEmbedder(dims=16)
    assert len(e.embed("test")) == 16


def test_embed_batch(embedder):
    results = embedder.embed_batch(["one", "two", "three"])
    assert len(results) == 3
    assert results[0] == embedder.embed("one")


def test_cosine_sim_identical():
    vec = [1.0, 0.0, 0.0]
    assert abs(cosine_sim(vec, vec) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(cosine_sim(a, b)) < 1e-6


def test_cosine_sim_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_sim(a, b) - (-1.0)) < 1e-6


def test_cosine_sim_zero_vector():
    assert cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0
