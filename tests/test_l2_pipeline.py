"""Phases 11-12: KL-gated L2 write path."""

from __future__ import annotations

from mnemo import InMemoryBackend, MemoryPolicy, Mnemo
from mnemo.embeddings.base import Embedder
from mnemo.extraction.l2_extractor import MockLLMExtractor


class _VanillaEmbedder(Embedder):
    def __init__(self) -> None:
        self._dim = 2

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if "vanilla" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_l2_not_called_when_kl_below_threshold() -> None:
    text = "I like vanilla updates only"
    llm = MockLLMExtractor({text: [("user", "note", "should not run")]})
    policy = MemoryPolicy(kl_threshold=0.99)
    emb = _VanillaEmbedder()
    mnemo = Mnemo(InMemoryBackend(), policy, embedder=emb, llm_extractor=llm)
    mnemo.semantic.store_fact("user", "bio", text, "user")
    _, result = mnemo.remember(text)
    assert result.write_level is None
    assert result.semantic_writes == []


def test_l2_called_when_l0_l1_miss_and_kl_high() -> None:
    text = "Quantum flux capacitor tolerance is 42 millivolts"
    llm = MockLLMExtractor({text: [("device", "tolerance", "42 millivolts")]})
    policy = MemoryPolicy(kl_threshold=0.05)
    mnemo = Mnemo(InMemoryBackend(), policy, llm_extractor=llm)
    _, result = mnemo.remember(text)
    assert result.write_level == 2
    assert len(result.semantic_writes) == 1
    assert result.semantic_writes[0].write_level == 2
