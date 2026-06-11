"""Temporal decay weights applied at retrieval time."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from mnemo.models import DecayMode, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy


def weight_exponential(age_days: float, half_life_days: float) -> float:
    """``e^(-λt)`` with ``λ = ln(2) / half_life`` so weight is 0.5 at half-life."""
    if half_life_days <= 0:
        return 0.0
    age = max(0.0, age_days)
    rate = math.log(2) / half_life_days
    return math.exp(-rate * age)


def weight_power_law(age_days: float, tau_days: float, alpha: float) -> float:
    """``(1 + t/τ)^(-α)`` — slower tail than exponential for typical α≈1."""
    if tau_days <= 0 or alpha <= 0:
        return 0.0
    age = max(0.0, age_days)
    return (1.0 + age / tau_days) ** (-alpha)


def decay_weight(
    age_days: float,
    mode: DecayMode,
    *,
    half_life_days: float = 30.0,
    tau_days: float = 7.0,
    alpha: float = 1.0,
) -> float:
    """Scalar weight in ``(0, 1]`` for a given age and policy curve."""
    if mode == DecayMode.NONE:
        return 1.0
    if mode == DecayMode.EXPONENTIAL:
        return weight_exponential(age_days, half_life_days)
    if mode == DecayMode.POWER_LAW:
        return weight_power_law(age_days, tau_days, alpha)
    raise ValueError(f"unknown decay mode: {mode!r}")


def _parse_iso(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def age_days_from_metadata(
    item: MemoryItem,
    reference: datetime,
    time_key: str,
) -> float:
    """Days between ``time_key`` in metadata and ``reference``."""
    raw = item.metadata.get(time_key)
    if raw is None:
        return 0.0
    stamp = _parse_iso(str(raw))
    ref = reference if reference.tzinfo else reference.replace(tzinfo=timezone.utc)
    delta = ref - stamp.astimezone(timezone.utc)
    return max(0.0, delta.total_seconds() / 86_400.0)


def decay_mode_for_tier(policy: MemoryPolicy, tier: MemoryTier) -> DecayMode:
    if tier == MemoryTier.EPISODIC:
        return policy.episodic_decay_mode
    if tier == MemoryTier.SEMANTIC:
        return policy.semantic_decay_mode
    return DecayMode.NONE


def decay_weight_for_item(
    item: MemoryItem,
    policy: MemoryPolicy,
    tier: MemoryTier,
    reference: datetime,
    *,
    time_key: str | None = None,
) -> float:
    """Retrieval weight for one item under tier-specific policy defaults."""
    mode = decay_mode_for_tier(policy, tier)
    if mode == DecayMode.NONE:
        return 1.0
    key = time_key or ("event_time" if tier == MemoryTier.EPISODIC else "valid_from")
    age = age_days_from_metadata(item, reference, key)
    return decay_weight(
        age,
        mode,
        half_life_days=policy.decay_half_life_days,
        tau_days=policy.decay_tau_days,
        alpha=policy.decay_alpha,
    )
