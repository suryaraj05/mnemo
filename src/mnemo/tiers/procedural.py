"""Tier 4 — Procedural memory: promoted patterns after N confirmed successes."""

from __future__ import annotations

import json
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.models import MemoryItem, MemoryTier


class ProceduralMemory:
    """Store action patterns once ``success_count >= min_successes`` (Phase 14)."""

    def __init__(self, backend: MemoryBackend, min_successes: int = 3) -> None:
        if min_successes < 1:
            raise ValueError("min_successes must be >= 1")
        self._backend = backend
        self._min_successes = min_successes
        self._episode_counter = 0

    def _next_episode_key(self) -> str:
        self._episode_counter += 1
        return f"proc_ep_{self._episode_counter}"

    def _find_pattern(self, task_type: str) -> MemoryItem | None:
        pattern_key = f"pattern:{task_type}"
        for item in self._backend.list(MemoryTier.PROCEDURAL, {}):
            if item.key == pattern_key:
                return item
        return None

    def record_episode(
        self,
        task_type: str,
        steps: list[str],
        success: bool,
        embedder: Embedder,
    ) -> str:
        """Log one run; promote pattern when successes reach threshold."""
        if not success:
            key = self._next_episode_key()
            self._backend.write(
                MemoryTier.PROCEDURAL,
                key,
                json.dumps(steps),
                {
                    "task_type": task_type,
                    "success": False,
                    "promoted": False,
                    "embedding": embedder.embed(task_type),
                },
            )
            return key

        pattern_key = f"pattern:{task_type}"
        existing = self._find_pattern(task_type)
        count = 1 if existing is None else int(existing.metadata.get("success_count", 0)) + 1
        promoted = count >= self._min_successes
        meta: dict[str, Any] = {
            "task_type": task_type,
            "success": True,
            "success_count": count,
            "promoted": promoted,
            "embedding": embedder.embed(task_type),
            "steps": steps,
        }
        self._backend.write(
            MemoryTier.PROCEDURAL,
            pattern_key,
            json.dumps(steps),
            meta,
        )
        return pattern_key

    def recall_patterns(
        self,
        task_query: str,
        embedder: Embedder,
        top_k: int,
    ) -> list[MemoryItem]:
        """Return promoted patterns ranked by cosine similarity to ``task_query``."""
        items = self._backend.list(MemoryTier.PROCEDURAL, {})
        promoted = [item for item in items if item.metadata.get("promoted")]
        if not promoted:
            return []

        query_vec = embedder.embed(task_query)
        scored: list[tuple[float, MemoryItem]] = []
        for item in promoted:
            embedding = item.metadata.get("embedding")
            if embedding is None:
                continue
            scored.append((cosine_similarity(query_vec, embedding), item))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [item for _, item in scored[:top_k]]
