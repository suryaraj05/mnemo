"""Composite tier-weighted recall (Phase 15)."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemo.backends.base import MemoryBackend
from mnemo.backends.vector_base import EMBEDDING_METADATA_KEY
from mnemo.decay import decay_weight_for_item
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.models import DecayMode, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy


def _keyword_score(query: str, item: MemoryItem) -> float:
    q = query.lower()
    if q in str(item.value).lower() or q in item.key.lower():
        return 1.0
    return 0.0


def composite_recall(
    backend: MemoryBackend,
    query: str,
    top_k: int,
    policy: MemoryPolicy,
    embedder: Embedder | None,
    *,
    reference_time: datetime | None = None,
) -> tuple[list[MemoryItem], list[MemoryTier]]:
    """Merge ranked hits from all tiers using hybrid scoring."""
    if top_k <= 0:
        return [], []

    ref = reference_time or datetime.now(timezone.utc)
    weights = policy.tier_weights
    query_vec = embedder.embed(query) if embedder is not None else None

    scored: list[tuple[float, MemoryItem, MemoryTier]] = []
    tiers_searched: list[MemoryTier] = []

    for tier in MemoryTier:
        tier_weight = weights.get(tier.value, 0.0)
        if tier_weight <= 0:
            continue
        tiers_searched.append(tier)
        items = backend.list(tier, {})

        if tier == MemoryTier.EPISODIC:
            items = [i for i in items if i.metadata.get("txn_to") is None]
        if tier == MemoryTier.SEMANTIC:
            items = [i for i in items if i.metadata.get("txn_to") is None]

        decay_mode = (
            policy.episodic_decay_mode
            if tier == MemoryTier.EPISODIC
            else policy.semantic_decay_mode
            if tier == MemoryTier.SEMANTIC
            else DecayMode.NONE
        )
        time_key = "event_time" if tier == MemoryTier.EPISODIC else "valid_from"

        for item in items:
            semantic = 0.0
            if query_vec is not None:
                embedding = item.metadata.get(EMBEDDING_METADATA_KEY)
                if embedding is not None:
                    semantic = max(0.0, cosine_similarity(query_vec, embedding))
            keyword = _keyword_score(query, item)
            temporal = (
                decay_weight_for_item(item, policy, tier, ref, time_key=time_key)
                if decay_mode != DecayMode.NONE
                else 1.0
            )
            score = tier_weight * (
                policy.weight_semantic * semantic
                + policy.weight_keyword * keyword
                + policy.weight_temporal * temporal
            )
            if score > 0:
                scored.append((score, item, tier))

    scored.sort(key=lambda row: row[0], reverse=True)
    seen: set[tuple[str, str]] = set()
    merged: list[MemoryItem] = []
    for _, item, tier in scored:
        identity = (tier.value, item.key)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
        if len(merged) >= top_k:
            break
    return merged, tiers_searched
