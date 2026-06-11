"""Vector similarity utilities for retrieval scoring."""

from __future__ import annotations

import math


def dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: list[float]) -> float:
    """Euclidean length of ``v``."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1].

    Returns ``0.0`` if either vector has zero norm (undefined angle guard).
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def normalize_l2(v: list[float]) -> list[float]:
    """Return unit vector; zero vector unchanged."""
    n = l2_norm(v)
    if n == 0.0:
        return list(v)
    return [x / n for x in v]
