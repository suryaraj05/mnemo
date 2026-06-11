"""Phase 4 tests: SQLiteBackend persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mnemo.backends.sqlite import SQLiteBackend
from mnemo.models import MemoryTier
from mnemo.tiers.episodic import EpisodicMemory


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def backend(db_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path)


def test_write_then_read(backend: SQLiteBackend) -> None:
    backend.write(MemoryTier.EPISODIC, "evt_1", "hello world", {"source": "user"})
    results = backend.read(MemoryTier.EPISODIC, "hello", top_k=5)
    assert len(results) == 1
    assert results[0].value == "hello world"


def test_persistence_across_connections(db_path: Path) -> None:
    b1 = SQLiteBackend(db_path)
    b1.write(MemoryTier.EPISODIC, "evt_1", "persist me", {"source": "user"})
    b1.close()

    b2 = SQLiteBackend(db_path)
    results = b2.read(MemoryTier.EPISODIC, "persist", top_k=5)
    b2.close()
    assert len(results) == 1


def test_delete_idempotent(backend: SQLiteBackend) -> None:
    backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
    backend.delete(MemoryTier.EPISODIC, "evt_1")
    backend.delete(MemoryTier.EPISODIC, "evt_1")
    assert backend.read(MemoryTier.EPISODIC, "hello", top_k=5) == []


def test_list_filters(backend: SQLiteBackend) -> None:
    backend.write(MemoryTier.EPISODIC, "a", "v1", {"source": "user"})
    backend.write(MemoryTier.EPISODIC, "b", "v2", {"source": "agent"})
    assert len(backend.list(MemoryTier.EPISODIC, {"source": "user"})) == 1


def test_episodic_survives_restart(db_path: Path) -> None:
    t0 = datetime(2026, 6, 4, 10, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 4, 10, 3, tzinfo=timezone.utc)

    em1 = EpisodicMemory(SQLiteBackend(db_path))
    em1.record("older", "user", event_time=t0)
    em1.record("newer", "user", event_time=t1)

    em2 = EpisodicMemory(SQLiteBackend(db_path))
    recent = em2.recall_recent(5)
    assert recent[0].value == "newer"
