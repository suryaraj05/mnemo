"""Abstract storage contract all Mnemo backends must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mnemo.models import MemoryItem, MemoryTier


class MemoryBackend(ABC):
    """Storage-agnostic contract for persisting and querying memory items.

    The interface is deliberately small and synchronous: backends store and
    retrieve, nothing more. Policy decisions (eviction, promotion, cost
    gating) and async orchestration live above this layer in the public API.

    Concrete implementations (in-memory, SQLite, hosted vector stores) are
    introduced in later phases against this frozen contract.
    """

    @abstractmethod
    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist ``value`` under ``key`` in ``tier``.

        Args:
            tier: Target memory tier.
            key: Unique key within the tier; writing an existing key replaces it.
            value: Arbitrary payload to store.
            metadata: Optional free-form metadata (timestamps, importance,
                confidence, provenance). ``None`` is equivalent to ``{}``.
        """

    @abstractmethod
    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        """Return up to ``top_k`` items from ``tier`` relevant to ``query``.

        Args:
            tier: Memory tier to search.
            query: Free-text query. Backends without semantic search may
                implement substring or recency-based matching.
            top_k: Maximum number of items to return, ranked most relevant first.

        Returns:
            At most ``top_k`` matching items; empty list when nothing matches.
        """

    @abstractmethod
    def delete(self, tier: MemoryTier, key: str) -> None:
        """Remove the item stored under ``key`` in ``tier``.

        Args:
            tier: Memory tier containing the item.
            key: Key of the item to remove. Deleting a missing key is a no-op;
                idempotency is required so retries are safe.
        """

    @abstractmethod
    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Enumerate items in ``tier``, optionally narrowed by metadata filters.

        Args:
            tier: Memory tier to enumerate.
            filters: Optional exact-match constraints applied to item metadata
                (e.g. ``{"entity": "user:42"}``). ``None`` returns everything.

        Returns:
            All items in the tier satisfying ``filters``.
        """
