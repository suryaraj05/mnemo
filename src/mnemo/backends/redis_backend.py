"""Redis-backed storage optimized for low-latency WORKING tier (Phase 17)."""

from __future__ import annotations

import json
from typing import Any

from mnemo.backends.base import MemoryBackend
from mnemo.models import MemoryItem, MemoryTier


class RedisBackend(MemoryBackend):
    """JSON-serialized items in Redis hashes per tier."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        ttl_seconds: int | None = None,
        key_prefix: str = "mnemo",
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                'RedisBackend requires redis: pip install "mnemo[redis]"'
            ) from exc
        self._client = redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _hash_key(self, tier: MemoryTier) -> str:
        return f"{self._prefix}:{tier.value}"

    def _serialize(self, item: MemoryItem) -> str:
        return json.dumps(item.model_dump())

    def _deserialize(self, raw: str) -> MemoryItem:
        return MemoryItem.model_validate(json.loads(raw))

    def write(
        self,
        tier: MemoryTier,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        item = MemoryItem(key=key, value=value, metadata=dict(metadata or {}))
        hash_key = self._hash_key(tier)
        self._client.hset(hash_key, key, self._serialize(item))
        if self._ttl is not None:
            self._client.expire(hash_key, self._ttl)

    def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
        if top_k <= 0:
            return []
        items = self.list(tier)
        hits = [
            item
            for item in items
            if query in item.key or query in str(item.value)
        ]
        return hits[:top_k]

    def delete(self, tier: MemoryTier, key: str) -> None:
        self._client.hdel(self._hash_key(tier), key)

    def list(
        self,
        tier: MemoryTier,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        raw = self._client.hgetall(self._hash_key(tier))
        items = [self._deserialize(value) for value in raw.values()]
        if not filters:
            return items
        return [item for item in items if all(item.metadata.get(k) == v for k, v in filters.items())]

    def close(self) -> None:
        self._client.close()
