"""Phase 9 integration: episodic recall with temporal decay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mnemo.backends.exact_vector import ExactVectorBackend
from mnemo.embeddings.base import Embedder
from mnemo.models import DecayMode
from mnemo.policy import MemoryPolicy
from mnemo.tiers.episodic import EpisodicMemory


class _VectorMapEmbedder(Embedder):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self._dim = len(next(iter(vectors.values())))

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return list(self._vectors[text])


def test_recall_semantic_decay_prefers_recent_similar_event() -> None:
    ref = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    old_time = ref - timedelta(days=60)
    new_time = ref - timedelta(days=1)

    query = "python learning"
    old_event = "I studied Python basics"
    new_event = "I studied Python basics recently"

    emb = _VectorMapEmbedder(
        {
            query: [1.0, 0.0],
            old_event: [0.99, 0.01],
            new_event: [0.98, 0.02],
        }
    )

    em = EpisodicMemory(ExactVectorBackend())
    em.record(old_event, "user", event_time=old_time, embedder=emb)
    em.record(new_event, "user", event_time=new_time, embedder=emb)

    policy = MemoryPolicy(
        episodic_decay_mode=DecayMode.EXPONENTIAL,
        decay_half_life_days=14.0,
    )
    hits = em.recall_semantic(query, emb, top_k=1, policy=policy, reference_time=ref)
    assert hits[0].value == new_event


def test_recall_semantic_without_decay_unchanged() -> None:
    ref = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    emb = _VectorMapEmbedder({"q": [1.0, 0.0], "a": [0.99, 0.01], "b": [0.98, 0.02]})
    em = EpisodicMemory(ExactVectorBackend())
    em.record("a", "user", event_time=ref - timedelta(days=1), embedder=emb)
    em.record("b", "user", event_time=ref - timedelta(days=60), embedder=emb)

    hits = em.recall_semantic("q", emb, top_k=1, reference_time=ref)
    assert hits[0].value == "a"
