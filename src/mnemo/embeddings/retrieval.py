"""Retrieval helpers combining embeddings with memory items."""

from __future__ import annotations

from typing import Any

from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY, VectorBackend
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.models import MemoryItem, MemoryTier


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
