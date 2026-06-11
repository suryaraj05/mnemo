"""Route tiers to different backend engines (e.g. Redis working + SQLite rest)."""

from __future__ import annotations

from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier


class CompositeBackend(MemoryBackend):
    """Delegate each tier to a dedicated backend implementation."""

    def __init__(
        self,
        routes: dict[MemoryTier, MemoryBackend],
        *,
        default: MemoryBackend | None = None,
    ) -> None:
        if not routes and default is None:
            raise ValueError("CompositeBackend needs routes and/or a default backend")
        self._routes = dict(routes)
        self._default = default

    def _pick(self, tier: MemoryTier) -> MemoryBackend:
        if tier in self._routes:
            return self._routes[tier]
        if self._default is not None:
            return self._default
        raise KeyError(f"no backend configured for tier {tier!r}")

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._pick(tier).write(tier, key, value, metadata)

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        return self._pick(tier).read(tier, query, top_k)

    def delete(self, tier: MemoryTier, key: str) -> None:
        self._pick(tier).delete(tier, key)

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return self._pick(tier).list(tier, filters)
