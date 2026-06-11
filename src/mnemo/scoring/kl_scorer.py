"""KL-based surprise estimator for L2 write-path gating (Phase 10)."""

from __future__ import annotations

import math

from mnemo.embeddings.similarity import cosine_similarity


def softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(temperature, 1e-9)
    scaled = [value / temp for value in values]
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """``KL(p || q)`` for discrete distributions."""
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > eps:
            total += pi * math.log((pi + eps) / (qi + eps))
    return max(0.0, total)


def _append_bin(dist: list[float], mass: float) -> list[float]:
    total = sum(dist) + mass
    if total <= 0:
        return [1.0]
    return [value / total for value in dist] + [mass / total]


def kl_surprise(
    utterance_vector: list[float],
    belief_vectors: list[list[float]],
    *,
    temperature: float = 0.05,
) -> float:
    """Estimate how surprising an utterance is vs existing semantic beliefs."""
    if not belief_vectors:
        return 1.0

    sims = [cosine_similarity(utterance_vector, belief) for belief in belief_vectors]
    if max(sims) >= 0.98:
        return 0.0

    q = _append_bin(softmax(sims, temperature), 0.05)
    novel = max(0.05, 1.0 - max(sims))
    p_new = _append_bin(softmax(sims + [novel * 2.0], temperature), novel)
    return kl_divergence(p_new, q)
