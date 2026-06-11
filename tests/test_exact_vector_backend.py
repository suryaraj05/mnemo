"""Phase 6 tests: ExactVectorBackend brute-force semantic search."""

from __future__ import annotations

from mnemo.backends.exact_vector import ExactVectorBackend
from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY
from mnemo.embeddings import HashEmbedder
from mnemo.models import MemoryTier


def test_search_by_vector_ranks_best_match() -> None:
    backend = ExactVectorBackend()
    emb = HashEmbedder(dimension=32)

    v_cat = emb.embed("the cat sat on the mat")
    v_dog = emb.embed("the dog chased a ball")
    v_query = emb.embed("kitten on a mat")

    backend.write(
        MemoryTier.EPISODIC,
        "e1",
        "cat event",
        {"embedding": v_cat, "source": "user"},
    )
    backend.write(
        MemoryTier.EPISODIC,
        "e2",
        "dog event",
        {"embedding": v_dog, "source": "user"},
    )

    results = backend.search_by_vector(MemoryTier.EPISODIC, v_query, top_k=1)
    assert len(results) == 1
    assert results[0].key == "e1"


def test_search_skips_items_without_embedding() -> None:
    backend = ExactVectorBackend()
    backend.write(MemoryTier.EPISODIC, "e1", "no vector", {})
    emb = HashEmbedder(16)
    results = backend.search_by_vector(MemoryTier.EPISODIC, emb.embed("x"), top_k=5)
    assert results == []


def test_search_respects_metadata_filters() -> None:
    backend = ExactVectorBackend()
    emb = HashEmbedder(16)
    v = emb.embed("hello")
    backend.write(MemoryTier.EPISODIC, "a", "A", {"embedding": v, "scope": "x"})
    backend.write(MemoryTier.EPISODIC, "b", "B", {"embedding": v, "scope": "y"})
    hits = backend.search_by_vector(
        MemoryTier.EPISODIC,
        v,
        top_k=5,
        filters={"scope": "x"},
    )
    assert len(hits) == 1
    assert hits[0].key == "a"
