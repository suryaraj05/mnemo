"""Phase 5 tests: similarity math and embedders."""

from __future__ import annotations

import math

import pytest

from mnemo.embeddings import HashEmbedder, cosine_similarity, dot, l2_norm, normalize_l2
from mnemo.embeddings.base import Embedder


def test_dot_product() -> None:
    assert dot([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert dot([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_identical_direction() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_guard() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


def test_normalize_l2() -> None:
    unit = normalize_l2([3.0, 4.0])
    assert l2_norm(unit) == pytest.approx(1.0)


def test_hash_embedder_deterministic() -> None:
    emb = HashEmbedder(dimension=32)
    a = emb.embed("hello world")
    b = emb.embed("hello world")
    assert a == b
    assert len(a) == 32


def test_hash_embedder_different_texts_differ() -> None:
    emb = HashEmbedder(dimension=32)
    assert emb.embed("alpha") != emb.embed("beta")


def test_embed_batch_default_loop() -> None:
    emb = HashEmbedder(dimension=16)
    batch = emb.embed_batch(["a", "b"])
    assert len(batch) == 2
    assert all(len(v) == 16 for v in batch)


def test_embedder_abc_dimension() -> None:
    emb = HashEmbedder(8)
    assert emb.dimension == 8


class _StubEmbedder(Embedder):
    def __init__(self) -> None:
        self._dim = 3

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


def test_custom_embedder_batch_override() -> None:
    stub = _StubEmbedder()
    vecs = stub.embed_batch(["hi", "hey"])
    assert len(vecs) == 2
    assert vecs[0][0] == 2.0
    assert vecs[1][0] == 3.0


def test_cosine_from_dot_product_identity() -> None:
    """cos(a,b) = a·b / (||a|| ||b||) for manual check."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    expected = dot(a, b) / (l2_norm(a) * l2_norm(b))
    assert cosine_similarity(a, b) == pytest.approx(expected)
    assert expected == pytest.approx(0.9746318, rel=1e-5)
