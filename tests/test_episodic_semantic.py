"""Episodic semantic recall integration tests."""

from __future__ import annotations

from mnemo.backends.exact_vector import ExactVectorBackend
from mnemo.embeddings import HashEmbedder
from mnemo.embeddings.base import Embedder
from mnemo.tiers.episodic import EpisodicMemory


class _VectorMapEmbedder(Embedder):
    """Returns fixed vectors per text — for ranking tests without ML."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self._dim = len(next(iter(vectors.values())))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return list(self._vectors[text])


def test_record_with_embedder_stores_vector() -> None:
    emb = HashEmbedder(32)
    em = EpisodicMemory(ExactVectorBackend())
    key = em.record("Paris is beautiful", "user", embedder=emb)
    item = em.recall_recent(1)[0]
    assert item.key == key
    assert "embedding" in item.metadata
    assert len(item.metadata["embedding"]) == 32


def test_recall_semantic_prefers_similar_event() -> None:
    python_event = "I love programming in Python"
    weather_event = "The weather is sunny today"
    query = "Python coding"
    emb = _VectorMapEmbedder(
        {
            python_event: [1.0, 0.0, 0.0],
            weather_event: [0.0, 1.0, 0.0],
            query: [0.9, 0.1, 0.0],
        }
    )
    em = EpisodicMemory(ExactVectorBackend())

    em.record(python_event, "user", embedder=emb)
    em.record(weather_event, "user", embedder=emb)

    hits = em.recall_semantic(query, emb, top_k=1)
    assert len(hits) == 1
    assert hits[0].value == python_event


def test_recall_semantic_excludes_retired() -> None:
    emb = HashEmbedder(32)
    em = EpisodicMemory(ExactVectorBackend())
    key = em.record("unique phrase alpha", "user", embedder=emb)
    em.record("unique phrase beta", "user", embedder=emb)
    em.retire(key)
    hits = em.recall_semantic("unique phrase alpha", emb, top_k=5)
    assert all(item.key != key for item in hits)
