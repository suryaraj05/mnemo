"""Brute-force cosine search — correct reference for CI (MemPalace sqlite_exact pattern)."""

from __future__ import annotations

from typing import Any

from mnemo.backends.memory import InMemoryBackend
from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY, VectorBackend
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.models import MemoryItem, MemoryTier


class ExactVectorBackend(VectorBackend):
    """In-memory storage with exact cosine ranking over stored embeddings."""

    def __init__(self) -> None:
        self._store = InMemoryBackend()

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store.write(tier, key, value, metadata)

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        return self._store.read(tier, query, top_k)

    def delete(self, tier: MemoryTier, key: str) -> None:
        self._store.delete(tier, key)

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return self._store.list(tier, filters)

    def search_by_vector(
        self,
        tier: MemoryTier,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if top_k <= 0:
            return []
        candidates = self._store.list(tier, filters or {})
        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates:
            embedding = item.metadata.get(EMBEDDING_METADATA_KEY)
            if embedding is None:
                continue
            score = cosine_similarity(query_vector, embedding)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k]]
