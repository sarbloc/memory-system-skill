"""Shared test fixtures."""

import pytest

from entity_memory.embedder import MockEmbedder


@pytest.fixture
def embedder():
    return MockEmbedder(dims=48)
