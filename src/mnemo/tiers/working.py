"""Tier 1 — Working memory: bounded context with importance-weighted eviction."""

from __future__ import annotations

import heapq
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier


class WorkingMemory:
    """Bounded WORKING-tier controller with min-heap eviction (ADR-003).

    Eviction key: lexicographic minimum of ``(importance, t, key)``.
    Policy B: when full, a new item is stored only if it beats the heap minimum.
    """

    def __init__(self, backend: MemoryBackend, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._backend = backend
        self._max_size = max_size
        self._heap: list[tuple[float, int, str]] = []
        # Canonical heap identity per key; stale heap tuples are skipped lazily.
        self._entries: dict[str, tuple[float, int]] = {}
        self._time = 0

    @classmethod
    def from_policy(cls, backend: MemoryBackend, policy_max_working_size: int) -> WorkingMemory:
        """Construct from :attr:`~mnemo.policy.MemoryPolicy.max_working_size`."""
        return cls(backend, policy_max_working_size)

    @staticmethod
    def _beats(s_a: float, t_a: int, s_b: float, t_b: int) -> bool:
        return s_a > s_b or (s_a == s_b and t_a > t_b)

    def _push(self, importance: float, t: int, key: str) -> None:
        self._entries[key] = (importance, t)
        heapq.heappush(self._heap, (importance, t, key))

    def _peek_min(self) -> tuple[float, int, str] | None:
        while self._heap:
            s, t, key = self._heap[0]
            if self._entries.get(key) == (s, t):
                return (s, t, key)
            heapq.heappop(self._heap)
        return None

    def _pop_min(self) -> tuple[float, int, str] | None:
        while self._heap:
            s, t, key = heapq.heappop(self._heap)
            if self._entries.get(key) == (s, t):
                del self._entries[key]
                return (s, t, key)
        return None

    def add(
        self,
        key: str,
        value: Any,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store one WORKING item. Returns False if rejected under Policy B."""
        self._time += 1
        t = self._time

        extra = dict(metadata) if metadata else {}
        meta = {
            "importance": importance,
            "t": t,
            "source": extra.pop("source", "user"),
            **extra,
        }

        # Overwrite existing key — does not consume a new slot.
        if key in self._entries:
            self._backend.write(MemoryTier.WORKING, key, value, meta)
            self._push(importance, t, key)
            return True

        if len(self._entries) < self._max_size:
            self._backend.write(MemoryTier.WORKING, key, value, meta)
            self._push(importance, t, key)
            return True

        minimum = self._peek_min()
        if minimum is None:
            self._backend.write(MemoryTier.WORKING, key, value, meta)
            self._push(importance, t, key)
            return True

        s_min, t_min, key_min = minimum
        if not self._beats(importance, t, s_min, t_min):
            return False

        self._pop_min()
        self._backend.delete(MemoryTier.WORKING, key_min)
        self._backend.write(MemoryTier.WORKING, key, value, meta)
        self._push(importance, t, key)
        return True

    def get_context(self) -> list[MemoryItem]:
        """Active WORKING items oldest-first by arrival counter ``t``."""
        items = self._backend.list(MemoryTier.WORKING, {})
        return sorted(items, key=lambda item: item.metadata["t"])

    def evict_one(self) -> str | None:
        """Evict the current heap minimum without inserting a replacement."""
        popped = self._pop_min()
        if popped is None:
            return None
        _, _, key = popped
        self._backend.delete(MemoryTier.WORKING, key)
        return key
