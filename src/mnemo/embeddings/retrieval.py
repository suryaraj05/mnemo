"""Retrieval helpers combining embeddings with memory items."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY, VectorBackend
from mnemo.decay import decay_weight_for_item
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.models import MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy


def rank_by_cosine(
    items: list[MemoryItem],
    query_vector: list[float],
    top_k: int,
) -> list[MemoryItem]:
    """Rank items with stored embeddings by cosine similarity (exact)."""
    if top_k <= 0:
        return []
    scored: list[tuple[float, MemoryItem]] = []
    for item in items:
        embedding = item.metadata.get(EMBEDDING_METADATA_KEY)
        if embedding is None:
            continue
        scored.append((cosine_similarity(query_vector, embedding), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def rank_by_cosine_with_decay(
    items: list[MemoryItem],
    query_vector: list[float],
    top_k: int,
    policy: MemoryPolicy,
    reference: datetime,
    *,
    tier: MemoryTier = MemoryTier.EPISODIC,
    time_key: str = "event_time",
) -> list[MemoryItem]:
    """Rank by ``cosine × decay_weight`` (retrieval-time forgetting)."""
    if top_k <= 0:
        return []
    scored: list[tuple[float, MemoryItem]] = []
    for item in items:
        embedding = item.metadata.get(EMBEDDING_METADATA_KEY)
        if embedding is None:
            continue
        cosine = cosine_similarity(query_vector, embedding)
        weight = decay_weight_for_item(item, policy, tier, reference, time_key=time_key)
        scored.append((cosine * weight, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def semantic_search(
    backend: Any,
    tier: MemoryTier,
    query_text: str,
    embedder: Embedder,
    top_k: int,
    filters: dict[str, Any] | None = None,
) -> list[MemoryItem]:
    """Embed query and search tier via VectorBackend or brute-force fallback."""
    query_vector = embedder.embed(query_text)
    if isinstance(backend, VectorBackend):
        return backend.search_by_vector(tier, query_vector, top_k, filters)
    items = backend.list(tier, filters or {})
    return rank_by_cosine(items, query_vector, top_k)
