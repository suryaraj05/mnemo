"""Vector search extension for semantic retrieval backends."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier

EMBEDDING_METADATA_KEY = "embedding"


class VectorBackend(MemoryBackend):
    """MemoryBackend plus approximate/exact nearest-neighbor search."""

    @abstractmethod
    def search_by_vector(
        self,
        tier: MemoryTier,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return up to ``top_k`` items ranked by cosine similarity to ``query_vector``."""
