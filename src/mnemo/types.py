"""Result types returned by Mnemo operations.

Every result carries cost and audit telemetry (``cost_usd``, ``latency_ms``,
timestamps) from day one, so later phases can add budget enforcement and
write-path observability without changing call-site signatures.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from mnemo.models import ForgetScope, MemoryItem, MemoryTier


def _utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


class WriteResult(BaseModel):
    """Outcome of writing one item to a memory tier.

    Attributes:
        tier: Tier the item was written to.
        key: Key the item was stored under.
        created: ``True`` if a new item was created, ``False`` on update.
        write_level: Cost-ladder level that produced the write
            (0 = regex, 1 = template embedding, 2 = LLM). ``None`` for
            direct writes that bypass extraction.
        latency_ms: End-to-end write latency in milliseconds.
        cost_usd: Marginal cost incurred by this write (e.g. LLM tokens).
        timestamp: UTC time the write completed.
    """

    tier: MemoryTier
    key: str
    created: bool = True
    write_level: int | None = None
    latency_ms: float | None = None
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=_utcnow)


class ReadResult(BaseModel):
    """Outcome of a recall/read operation.

    Attributes:
        query: The query string that was executed.
        items: Retrieved items, ranked most relevant first.
        tiers_searched: Tiers consulted to produce ``items``.
        top_k: Requested maximum number of items, if specified.
        latency_ms: End-to-end read latency in milliseconds.
        cost_usd: Marginal cost incurred (e.g. query embedding).
        timestamp: UTC time the read completed.
    """

    query: str
    items: list[MemoryItem] = Field(default_factory=list)
    tiers_searched: list[MemoryTier] = Field(default_factory=list)
    top_k: int | None = None
    latency_ms: float | None = None
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=_utcnow)


class DeleteResult(BaseModel):
    """Outcome of a forget/delete operation.

    Attributes:
        scope: Granularity of the deletion that was performed.
        tier: Tier affected, when ``scope`` targets a single tier or item.
        key: Key removed, when ``scope`` is :attr:`ForgetScope.ITEM`.
        deleted_count: Number of items actually removed (audit trail).
        latency_ms: End-to-end delete latency in milliseconds.
        timestamp: UTC time the deletion completed.
    """

    scope: ForgetScope = ForgetScope.ITEM
    tier: MemoryTier | None = None
    key: str | None = None
    deleted_count: int = 0
    latency_ms: float | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
