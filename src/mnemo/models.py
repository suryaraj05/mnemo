"""Core data models for Mnemo.

Defines the memory-tier taxonomy, forget scopes, and the canonical
:class:`MemoryItem` record exchanged between backends and the public API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """The four memory tiers Mnemo manages.

    Attributes:
        WORKING: Bounded in-context memory with importance-weighted FIFO
            eviction (min-heap on ``(importance, arrival_time, key)``).
        EPISODIC: Verbatim, timestamped events with bi-temporal metadata;
            never summarized at write time.
        SEMANTIC: Extracted facts with validity windows and confidence scores.
        PROCEDURAL: Behavioral patterns promoted after N confirmed successes.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ForgetScope(str, Enum):
    """Granularity of a forget/delete operation.

    Attributes:
        ITEM: Remove a single memory item by key.
        TIER: Clear an entire tier.
        ALL: Clear every tier.
        ENTITY: Remove all memories referencing an entity across all tiers
            (reserved for future GDPR-style erasure).
    """

    ITEM = "item"
    TIER = "tier"
    ALL = "all"
    ENTITY = "entity"


class MemoryItem(BaseModel):
    """A single unit of stored memory, tier-agnostic.

    Attributes:
        key: Unique identifier of the item within its tier.
        value: Arbitrary payload (text, structured dict, etc.).
        metadata: Free-form metadata such as timestamps, importance,
            confidence, validity windows, or provenance. Tier-specific
            conventions are layered on top in later phases.
    """

    key: str
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
