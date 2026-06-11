"""Dict-backed in-memory backend: the executable specification of ADR-002."""

from __future__ import annotations

from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier


class InMemoryBackend(MemoryBackend):
    """Non-persistent reference backend storing items in nested dicts.

    Semantics (pinned in ADR-002):

    - ``write`` upserts on ``(tier, key)``; ``metadata=None`` is stored as ``{}``;
      a shallow copy of metadata is stored so caller mutation cannot leak in.
      Overwriting keeps the item's original insertion position.
    - ``read`` is a case-sensitive substring match on ``item.key`` or
      ``str(item.value)``, returned in insertion order and sliced to ``top_k``.
      Missing tier or ``top_k <= 0`` returns ``[]``.
    - ``delete`` is idempotent: missing tier or key is a silent no-op.
    - ``list`` returns items where every filter pair equals
      ``item.metadata.get(k)``; ``None`` and ``{}`` both mean "return all".

    Not thread-safe and not persistent: intended for tests, notebooks, and
    single-threaded agents — it is the spec and the test double, not a
    production store.
    """

    def __init__(self) -> None:
        """Create an empty store with no tier buckets allocated."""
        self._store: dict[MemoryTier, dict[str, MemoryItem]] = {}

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upsert ``value`` under ``key`` in ``tier`` (ADR-002 write rules)."""
        bucket = self._store.setdefault(tier, {})
        bucket[key] = MemoryItem(
            key=key,
            value=value,
            metadata=dict(metadata) if metadata else {},
        )

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        """Return up to ``top_k`` substring matches in insertion order."""
        bucket = self._store.get(tier)
        if not bucket or top_k <= 0:
            return []
        matches = [
            item
            for item in bucket.values()
            if query in item.key or query in str(item.value)
        ]
        return matches[:top_k]

    def delete(self, tier: MemoryTier, key: str) -> None:
        """Remove ``key`` from ``tier``; silent no-op when either is missing."""
        bucket = self._store.get(tier)
        if bucket is not None:
            bucket.pop(key, None)

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return items in ``tier`` whose metadata exactly matches ``filters``."""
        bucket = self._store.get(tier)
        if not bucket:
            return []
        active = filters or {}
        return [
            item
            for item in bucket.values()
            if all(item.metadata.get(k) == v for k, v in active.items())
        ]
