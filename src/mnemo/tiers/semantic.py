"""Tier 3 — Semantic memory: extracted facts with bi-temporal validity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.extraction.l0_extractor import extract_l0
from mnemo.extraction.l1_extractor import extract_l1
from mnemo.extraction.models import ExtractedFact
from mnemo.extraction.templates import TemplateLibrary, load_template_library
from mnemo.models import MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy
from mnemo.types import WriteResult


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SemanticMemory:
    """Entity–predicate facts on SEMANTIC tier with conflict resolution (ADR-007)."""

    def __init__(self, backend: MemoryBackend) -> None:
        self._backend = backend
        self._counter = 0

    def _next_key(self) -> str:
        self._counter += 1
        return f"fact_{self._counter}"

    def _active_items(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        items = self._backend.list(MemoryTier.SEMANTIC, filters or {})
        return [item for item in items if item.metadata.get("txn_to") is None]

    def _find_active(self, entity: str, predicate: str) -> MemoryItem | None:
        for item in self._active_items({"entity": entity, "predicate": predicate}):
            if item.metadata.get("entity") == entity and item.metadata.get("predicate") == predicate:
                return item
        return None

    def _close_row(self, item: MemoryItem, when: datetime) -> None:
        new_meta = dict(item.metadata)
        new_meta["txn_to"] = when.isoformat()
        if new_meta.get("valid_to") is None:
            new_meta["valid_to"] = when.isoformat()
        self._backend.write(MemoryTier.SEMANTIC, item.key, item.value, new_meta)

    def store_fact(
        self,
        entity: str,
        predicate: str,
        value: str,
        source: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        write_level: int | None = None,
        cost_usd: float = 0.0,
    ) -> WriteResult:
        """Upsert one fact; closes the prior active row on value change."""
        when = _utc_now()
        existing = self._find_active(entity, predicate)
        created = True

        if existing is not None:
            if existing.value == value:
                return WriteResult(
                    tier=MemoryTier.SEMANTIC,
                    key=existing.key,
                    created=False,
                    write_level=write_level,
                    cost_usd=cost_usd,
                    timestamp=when,
                )
            self._close_row(existing, when)
            created = True

        valid_start = valid_from or when
        meta: dict[str, Any] = {
            "entity": entity,
            "predicate": predicate,
            "valid_from": valid_start.isoformat(),
            "valid_to": None,
            "txn_from": when.isoformat(),
            "txn_to": None,
            "confidence": confidence,
            "source": source,
        }
        if write_level is not None:
            meta["write_level"] = write_level

        key = self._next_key()
        self._backend.write(MemoryTier.SEMANTIC, key, value, meta)
        return WriteResult(
            tier=MemoryTier.SEMANTIC,
            key=key,
            created=created,
            write_level=write_level,
            cost_usd=cost_usd,
            timestamp=when,
        )

    def ingest_l0(self, text: str, *, default_entity: str = "user") -> list[WriteResult]:
        """Run L0 regex extraction on ``text`` and store each fact."""
        results: list[WriteResult] = []
        for fact in extract_l0(text, default_entity=default_entity):
            results.append(self._store_extracted(fact, source="l0:regex"))
        return results

    def _store_extracted(
        self,
        fact: ExtractedFact,
        source: str,
        *,
        cost_usd: float = 0.0,
    ) -> WriteResult:
        return self.store_fact(
            fact.entity,
            fact.predicate,
            fact.value,
            source,
            confidence=fact.confidence,
            valid_from=fact.valid_from,
            write_level=fact.write_level,
            cost_usd=cost_usd,
        )

    def ingest_l1(
        self,
        text: str,
        embedder: Embedder,
        library: TemplateLibrary | None = None,
        *,
        policy: MemoryPolicy | None = None,
        default_entity: str = "user",
    ) -> list[WriteResult]:
        """Run L1 template embedding match on ``text`` and store matched facts."""
        lib = library or load_template_library()
        cfg = policy or MemoryPolicy()
        batch = extract_l1(
            text,
            embedder,
            lib,
            policy=cfg,
            default_entity=default_entity,
        )
        if not batch.facts:
            return []
        share = batch.cost_usd / len(batch.facts)
        return [
            self._store_extracted(fact, "l1:template", cost_usd=share)
            for fact in batch.facts
        ]

    def get_fact(self, entity: str, predicate: str) -> MemoryItem | None:
        """Return the active fact for ``entity`` + ``predicate``, if any."""
        return self._find_active(entity, predicate)

    def recall_facts(
        self,
        *,
        entity: str | None = None,
        predicate: str | None = None,
    ) -> list[MemoryItem]:
        """Active facts, optionally filtered by entity and/or predicate."""
        filters: dict[str, Any] = {}
        if entity is not None:
            filters["entity"] = entity
        if predicate is not None:
            filters["predicate"] = predicate
        active = self._active_items(filters or None)
        if entity is not None:
            active = [i for i in active if i.metadata.get("entity") == entity]
        if predicate is not None:
            active = [i for i in active if i.metadata.get("predicate") == predicate]
        return sorted(active, key=lambda i: i.metadata["txn_from"])

    def get_history(self, entity: str, predicate: str) -> list[MemoryItem]:
        """All rows for a logical fact, oldest transaction first (audit trail)."""
        rows = [
            item
            for item in self._backend.list(MemoryTier.SEMANTIC, {})
            if item.metadata.get("entity") == entity and item.metadata.get("predicate") == predicate
        ]
        return sorted(rows, key=lambda i: i.metadata["txn_from"])
