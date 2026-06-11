"""Phase 2 tests: WorkingMemory eviction controller."""

from __future__ import annotations

import pytest

from mnemo.backends.memory import InMemoryBackend
from mnemo.models import MemoryTier
from mnemo.policy import MemoryPolicy
from mnemo.tiers.working import WorkingMemory


@pytest.fixture
def wm() -> WorkingMemory:
    return WorkingMemory(InMemoryBackend(), max_size=2)


def test_add_under_capacity(wm: WorkingMemory) -> None:
    assert wm.add("a", "A", 0.9) is True
    assert wm.add("b", "B", 0.1) is True
    assert len(wm.get_context()) == 2


def test_evict_lowest_importance(wm: WorkingMemory) -> None:
    wm.add("a", "A", 0.9)
    wm.add("b", "B", 0.1)
    assert wm.add("c", "C", 0.5) is True
    keys = {item.key for item in wm.get_context()}
    assert keys == {"a", "c"}


def test_equal_importance_evicts_older(wm: WorkingMemory) -> None:
    wm.add("a", "A", 0.5)
    wm.add("b", "B", 0.3)
    assert wm.add("c", "C", 0.3) is True
    keys = {item.key for item in wm.get_context()}
    assert keys == {"a", "c"}


def test_reject_when_not_better(wm: WorkingMemory) -> None:
    wm.add("a", "A", 0.9)
    wm.add("b", "B", 0.8)
    assert wm.add("c", "C", 0.1) is False
    assert len(wm.get_context()) == 2


def test_get_context_oldest_first(wm: WorkingMemory) -> None:
    wm.add("x", "third", 0.5)
    wm.add("y", "first", 0.5)
    ctx = wm.get_context()
    assert [i.metadata["t"] for i in ctx] == sorted(i.metadata["t"] for i in ctx)


def test_evict_one(wm: WorkingMemory) -> None:
    wm.add("a", "A", 0.9)
    wm.add("b", "B", 0.1)
    evicted = wm.evict_one()
    assert evicted == "b"
    assert len(wm.get_context()) == 1


def test_metadata_none_safe() -> None:
    wm = WorkingMemory(InMemoryBackend(), max_size=2)
    assert wm.add("solo", "hello", 0.7, None) is True


def test_max_size_one() -> None:
    wm = WorkingMemory(InMemoryBackend(), max_size=1)
    wm.add("a", "A", 0.5)
    assert wm.add("b", "B", 0.9) is True
    assert len(wm.get_context()) == 1
    assert wm.get_context()[0].key == "b"


def test_same_key_overwrite_one_heap_slot(wm: WorkingMemory) -> None:
    wm.add("turn_1", "hello", 0.9)
    wm.add("turn_1", "hello UPDATED", 0.8)
    assert len(wm.get_context()) == 1
    assert wm.get_context()[0].value == "hello UPDATED"


def test_rejected_add_still_increments_t(wm: WorkingMemory) -> None:
    wm.add("a", "A", 0.9)
    wm.add("b", "B", 0.8)
    wm.add("c", "C", 0.1)
    assert wm.add("d", "D", 0.85) is True
    ts = [i.metadata["t"] for i in wm.get_context()]
    assert 3 not in ts


def test_working_tier_isolated(wm: WorkingMemory) -> None:
    wm._backend.write(MemoryTier.EPISODIC, "e1", "secret", {})  # type: ignore[attr-defined]
    wm.add("a", "A", 0.5)
    assert len(wm.get_context()) == 1


def test_from_policy() -> None:
    policy = MemoryPolicy(max_working_size=3)
    wm = WorkingMemory.from_policy(InMemoryBackend(), policy.max_working_size)
    for i in range(3):
        wm.add(f"k{i}", f"v{i}", 0.5)
    assert len(wm.get_context()) == 3


def test_invalid_max_size() -> None:
    with pytest.raises(ValueError):
        WorkingMemory(InMemoryBackend(), max_size=0)
