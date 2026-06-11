"""Phase 3 tests: EpisodicMemory verbatim + bi-temporal."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemo.backends.memory import InMemoryBackend
from mnemo.tiers.episodic import EpisodicMemory


@pytest.fixture
def em() -> EpisodicMemory:
    return EpisodicMemory(InMemoryBackend())


def test_record_verbatim(em: EpisodicMemory) -> None:
    key = em.record("I moved to Pune last month.", "user")
    assert key.startswith("evt_")
    recent = em.recall_recent(5)
    assert len(recent) == 1
    assert recent[0].value == "I moved to Pune last month."


def test_recall_recent_newest_event_time_first(em: EpisodicMemory) -> None:
    t0 = datetime(2026, 6, 4, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 4, 10, 3, tzinfo=timezone.utc)
    em.record("older", "user", event_time=t0)
    em.record("newer", "user", event_time=t1)
    recent = em.recall_recent(5)
    assert recent[0].value == "newer"
    assert recent[1].value == "older"


def test_retire_hides_from_recall(em: EpisodicMemory) -> None:
    key = em.record("to retire", "user")
    em.retire(key)
    assert em.recall_recent(5) == []


def test_get_timeline_oldest_first(em: EpisodicMemory) -> None:
    t0 = datetime(2026, 6, 4, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 4, 10, 3, tzinfo=timezone.utc)
    em.record("first", "user", event_time=t0)
    em.record("second", "user", event_time=t1)
    timeline = em.get_timeline()
    assert timeline[0].value == "first"
    assert timeline[-1].value == "second"


def test_retire_missing_key_raises(em: EpisodicMemory) -> None:
    with pytest.raises(KeyError):
        em.retire("evt_missing")


def test_scope_in_metadata(em: EpisodicMemory) -> None:
    em.record("hello", "user", scope="voxgraph")
    item = em.recall_recent(1)[0]
    assert item.metadata["scope"] == "voxgraph"
