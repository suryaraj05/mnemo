"""Tier 2 — Episodic memory: verbatim events with bi-temporal metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY, VectorBackend
from mnemo.decay import decay_weight_for_item
from mnemo.models import DecayMode
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.retrieval import rank_by_cosine, rank_by_cosine_with_decay
from mnemo.models import MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy


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
        embedder: Embedder | None = None,
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
        if embedder is not None:
            meta[EMBEDDING_METADATA_KEY] = embedder.embed(event_text)

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
        *,
        policy: MemoryPolicy | None = None,
        reference_time: datetime | None = None,
    ) -> list[MemoryItem]:
        """Active events only, ranked by ``event_time`` (optionally decay-weighted)."""
        active = self._active_items(filters)
        cfg = policy or MemoryPolicy()
        ref = reference_time or _utc_now()
        if cfg.episodic_decay_mode == DecayMode.NONE:
            ranked = sorted(active, key=lambda i: i.metadata["event_time"], reverse=True)
            return ranked[:top_k]
        scored = [
            (
                decay_weight_for_item(item, cfg, MemoryTier.EPISODIC, ref, time_key="event_time"),
                item.metadata["event_time"],
                item,
            )
            for item in active
        ]
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _, _, item in scored[:top_k]]

    def recall_semantic(
        self,
        query: str,
        embedder: Embedder,
        top_k: int,
        filters: dict[str, Any] | None = None,
        *,
        policy: MemoryPolicy | None = None,
        reference_time: datetime | None = None,
    ) -> list[MemoryItem]:
        """Active events ranked by cosine similarity to ``query`` embedding."""
        cfg = policy or MemoryPolicy()
        ref = reference_time or _utc_now()
        overfetch = max(top_k * 3, top_k)
        query_vector = embedder.embed(query)

        if isinstance(self._backend, VectorBackend):
            hits = self._backend.search_by_vector(
                MemoryTier.EPISODIC,
                query_vector,
                overfetch,
                filters,
            )
        else:
            items = self._backend.list(MemoryTier.EPISODIC, filters or {})
            hits = rank_by_cosine(items, query_vector, overfetch)

        active = [item for item in hits if item.metadata.get("txn_to") is None]
        if cfg.episodic_decay_mode == DecayMode.NONE:
            return active[:top_k]
        return rank_by_cosine_with_decay(
            active,
            query_vector,
            top_k,
            cfg,
            ref,
            tier=MemoryTier.EPISODIC,
            time_key="event_time",
        )

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
