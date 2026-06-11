"""Cross-tier forget / GDPR erasure (Phase 16)."""

from __future__ import annotations

import time

from mnemo.backends.base import MemoryBackend
from mnemo.models import ForgetScope, MemoryTier
from mnemo.types import DeleteResult


def forget(
    backend: MemoryBackend,
    scope: ForgetScope,
    *,
    tier: MemoryTier | None = None,
    key: str | None = None,
    entity: str | None = None,
) -> DeleteResult:
    """Delete items according to ``ForgetScope``."""
    start = time.perf_counter()
    deleted = 0

    if scope == ForgetScope.ITEM:
        if tier is None or key is None:
            raise ValueError("ITEM scope requires tier and key")
        existed = any(item.key == key for item in backend.list(tier, {}))
        backend.delete(tier, key)
        deleted = 1 if existed else 0

    elif scope == ForgetScope.TIER:
        if tier is None:
            raise ValueError("TIER scope requires tier")
        items = backend.list(tier, {})
        for item in items:
            backend.delete(tier, item.key)
        deleted = len(items)

    elif scope == ForgetScope.ALL:
        for mem_tier in MemoryTier:
            items = backend.list(mem_tier, {})
            for item in items:
                backend.delete(mem_tier, item.key)
            deleted += len(items)

    elif scope == ForgetScope.ENTITY:
        if entity is None:
            raise ValueError("ENTITY scope requires entity")
        for mem_tier in MemoryTier:
            for item in backend.list(mem_tier, {}):
                if item.metadata.get("entity") == entity:
                    backend.delete(mem_tier, item.key)
                    deleted += 1
    else:
        raise ValueError(f"unsupported scope: {scope!r}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return DeleteResult(
        scope=scope,
        tier=tier,
        key=key,
        deleted_count=deleted,
        latency_ms=latency_ms,
    )
