"""Composite backend routing."""

from __future__ import annotations

from mnemo.backends.composite import CompositeBackend
from mnemo.backends.memory import InMemoryBackend
from mnemo.models import MemoryTier


def test_routes_tiers_to_dedicated_backends() -> None:
    working = InMemoryBackend()
    long_term = InMemoryBackend()
    composite = CompositeBackend(
        {MemoryTier.WORKING: working},
        default=long_term,
    )
    composite.write(MemoryTier.WORKING, "w1", "fast", {})
    composite.write(MemoryTier.EPISODIC, "e1", "slow", {})
    assert working.list(MemoryTier.WORKING, {})
    assert long_term.list(MemoryTier.EPISODIC, {})
