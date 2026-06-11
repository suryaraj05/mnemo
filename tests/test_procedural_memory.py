"""Phase 14: Procedural memory."""

from __future__ import annotations

from mnemo.backends.memory import InMemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.tiers.procedural import ProceduralMemory


class _StubEmbedder(Embedder):
    def __init__(self) -> None:
        self._dim = 3

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


def test_pattern_promoted_after_n_successes() -> None:
    pm = ProceduralMemory(InMemoryBackend(), min_successes=3)
    emb = _StubEmbedder()
    steps = ["open", "edit", "save"]
    for _ in range(2):
        pm.record_episode("deploy", steps, success=True, embedder=emb)
    assert pm.recall_patterns("deploy", emb, top_k=1) == []
    pm.record_episode("deploy", steps, success=True, embedder=emb)
    patterns = pm.recall_patterns("deploy", emb, top_k=1)
    assert len(patterns) == 1
    assert patterns[0].metadata["success_count"] == 3


def test_failure_does_not_promote() -> None:
    pm = ProceduralMemory(InMemoryBackend(), min_successes=2)
    emb = _StubEmbedder()
    pm.record_episode("deploy", ["fail"], success=False, embedder=emb)
    pm.record_episode("deploy", ["ok"], success=True, embedder=emb)
    assert pm.recall_patterns("deploy", emb, top_k=1) == []
