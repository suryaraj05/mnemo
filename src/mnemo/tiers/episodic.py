"""Tier 2 — Episodic memory: verbatim events with bi-temporal metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EpisodicMemory:
    """Append-only verbatim log on EPISODIC tier (ADR-004). Never summarize at write."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend
        self._counter = 0

    def _next_key(self) -> str:
        self._counter += 1
        return f"evt_{self._counter}"

    def record(
        self,
        event_text: str,
        source: str,
        event_time: datetime | None = None,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one raw event verbatim. Returns the generated event key."""
        when = event_time or _utc_now()
        meta: dict[str, Any] = {
            "event_time": when.isoformat(),
            "txn_from": _utc_now().isoformat(),
            "txn_to": None,
            "source": source,
        }
        if scope is not None:
            meta["scope"] = scope
        if metadata:
            meta.update(metadata)

        key = self._next_key()
        self._backend.write(MemoryTier.EPISODIC, key, event_text, meta)
        return key

    def _active_items(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        items = self._backend.list(MemoryTier.EPISODIC, filters or {})
        return [item for item in items if item.metadata.get("txn_to") is None]

    def recall_recent(
        self,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Active events only, newest ``event_time`` first."""
        active = self._active_items(filters)
        ranked = sorted(active, key=lambda i: i.metadata["event_time"], reverse=True)
        return ranked[:top_k]

    def get_timeline(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Active events oldest-first (conversation replay order)."""
        active = self._active_items(filters)
        return sorted(active, key=lambda i: i.metadata["event_time"])

    def retire(self, key: str) -> None:
        """Close ``txn_to`` — row stays for audit but is hidden from recall."""
        matches = [i for i in self._backend.list(MemoryTier.EPISODIC, {}) if i.key == key]
        if not matches:
            raise KeyError(f"episodic key not found: {key!r}")
        item = matches[0]
        new_meta = dict(item.metadata)
        new_meta["txn_to"] = _utc_now().isoformat()
        self._backend.write(MemoryTier.EPISODIC, item.key, item.value, new_meta)
