"""Phase 8 integration: SemanticMemory.ingest_l1."""

from __future__ import annotations

import pytest

from mnemo.backends.memory import InMemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.extraction.templates import FactTemplate, TemplateLibrary
from mnemo.policy import MemoryPolicy
from mnemo.tiers.semantic import SemanticMemory


class _ClusterEmbedder(Embedder):
    def __init__(self) -> None:
        self._dim = 3

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        low = text.lower()
        if "started at" in low or "work at" in low:
            return [1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
def library() -> TemplateLibrary:
    return TemplateLibrary(
        templates=[
            FactTemplate(
                entity="user",
                predicate="employer",
                utterances=["I work at Acme Corp"],
                value_pattern=r"(?i)(?:work at|started at)\s+([A-Za-z]+)",
            )
        ]
    )


def test_ingest_l1_stores_fact_with_write_level(library: TemplateLibrary) -> None:
    sm = SemanticMemory(InMemoryBackend())
    policy = MemoryPolicy(l1_cosine_threshold=0.5)
    results = sm.ingest_l1("I started at Figma", _ClusterEmbedder(), library, policy=policy)
    assert len(results) == 1
    assert results[0].write_level == 1
    assert results[0].created is True
    fact = sm.get_fact("user", "employer")
    assert fact is not None
    assert fact.value == "Figma"
    assert fact.metadata["source"] == "l1:template"


def test_ingest_l1_write_result_cost(library: TemplateLibrary) -> None:
    sm = SemanticMemory(InMemoryBackend())
    policy = MemoryPolicy(l1_embed_cost_usd=0.005, l1_cosine_threshold=0.5)
    results = sm.ingest_l1(
        "I work at Acme Corp",
        _ClusterEmbedder(),
        library,
        policy=policy,
    )
    assert results[0].cost_usd == pytest.approx(0.005)
