"""Contract tests for InMemoryBackend per ADR-002."""

from __future__ import annotations

import pytest

from mnemo.backends import InMemoryBackend
from mnemo.models import MemoryTier


@pytest.fixture()
def backend() -> InMemoryBackend:
    """Fresh, empty backend per test."""
    return InMemoryBackend()


class TestHappyPath:
    def test_write_then_read_substring_on_value(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello world", {"source": "user"})
        items = backend.read(MemoryTier.EPISODIC, "hello", top_k=5)
        assert len(items) == 1
        assert items[0].key == "evt_1"
        assert items[0].value == "hello world"
        assert items[0].metadata == {"source": "user"}

    def test_delete_existing_then_read_empty(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
        backend.delete(MemoryTier.EPISODIC, "evt_1")
        assert backend.read(MemoryTier.EPISODIC, "hello", top_k=5) == []

    def test_tiers_are_isolated(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
        assert backend.list(MemoryTier.SEMANTIC) == []
        assert backend.read(MemoryTier.SEMANTIC, "hello", top_k=5) == []
        assert len(backend.list(MemoryTier.EPISODIC)) == 1

    def test_list_filters_metadata(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "a", "x", {"source": "user", "scope": "s1"})
        backend.write(MemoryTier.EPISODIC, "b", "y", {"source": "agent", "scope": "s1"})
        backend.write(MemoryTier.EPISODIC, "c", "z", {"source": "user", "scope": "s2"})
        keys = [i.key for i in backend.list(MemoryTier.EPISODIC, {"source": "user"})]
        assert keys == ["a", "c"]
        keys = [
            i.key
            for i in backend.list(MemoryTier.EPISODIC, {"source": "user", "scope": "s1"})
        ]
        assert keys == ["a"]


class TestEdgeCases:
    def test_read_missing_tier_returns_empty(self, backend: InMemoryBackend) -> None:
        assert backend.read(MemoryTier.PROCEDURAL, "anything", top_k=3) == []

    def test_list_filters_none_and_empty_return_all(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.WORKING, "a", 1, {"importance": 0.5})
        backend.write(MemoryTier.WORKING, "b", 2)
        assert len(backend.list(MemoryTier.WORKING, filters=None)) == 2
        assert len(backend.list(MemoryTier.WORKING, filters={})) == 2

    def test_read_respects_top_k_and_insertion_order(self, backend: InMemoryBackend) -> None:
        for i in range(5):
            backend.write(MemoryTier.EPISODIC, f"evt_{i}", "common payload")
        items = backend.read(MemoryTier.EPISODIC, "common", top_k=3)
        assert [i.key for i in items] == ["evt_0", "evt_1", "evt_2"]

    def test_write_overwrites_same_key(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.SEMANTIC, "fact", "old", {"confidence": 0.4})
        backend.write(MemoryTier.SEMANTIC, "fact", "new", {"confidence": 0.9})
        items = backend.list(MemoryTier.SEMANTIC)
        assert len(items) == 1
        assert items[0].value == "new"
        assert items[0].metadata == {"confidence": 0.9}

    def test_read_top_k_zero_returns_empty(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
        assert backend.read(MemoryTier.EPISODIC, "hello", top_k=0) == []

    def test_read_is_case_sensitive(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "Hello World")
        assert backend.read(MemoryTier.EPISODIC, "hello", top_k=5) == []
        assert len(backend.read(MemoryTier.EPISODIC, "Hello", top_k=5)) == 1


class TestDeleteIdempotency:
    def test_delete_missing_tier_no_raise(self, backend: InMemoryBackend) -> None:
        backend.delete(MemoryTier.PROCEDURAL, "ghost")  # must not raise

    def test_delete_missing_key_no_raise(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
        backend.delete(MemoryTier.EPISODIC, "ghost")  # must not raise
        assert len(backend.list(MemoryTier.EPISODIC)) == 1

    def test_delete_twice_no_raise(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "evt_1", "hello")
        backend.delete(MemoryTier.EPISODIC, "evt_1")
        backend.delete(MemoryTier.EPISODIC, "evt_1")  # idempotent
        assert backend.list(MemoryTier.EPISODIC) == []


class TestExtras:
    def test_read_matches_key_substring(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.EPISODIC, "meeting_2026", 12345)
        items = backend.read(MemoryTier.EPISODIC, "meeting", top_k=5)
        assert [i.key for i in items] == ["meeting_2026"]

    def test_list_filter_missing_metadata_key_excludes_item(
        self, backend: InMemoryBackend
    ) -> None:
        backend.write(MemoryTier.EPISODIC, "a", "x")  # no 'source' key
        backend.write(MemoryTier.EPISODIC, "b", "y", {"source": "user"})
        keys = [i.key for i in backend.list(MemoryTier.EPISODIC, {"source": "user"})]
        assert keys == ["b"]

    def test_metadata_none_stored_as_empty_dict(self, backend: InMemoryBackend) -> None:
        backend.write(MemoryTier.WORKING, "k", "v", metadata=None)
        assert backend.list(MemoryTier.WORKING)[0].metadata == {}

    def test_metadata_not_aliased_to_caller_dict(self, backend: InMemoryBackend) -> None:
        meta = {"source": "user"}
        backend.write(MemoryTier.WORKING, "k", "v", metadata=meta)
        meta["source"] = "tampered"
        assert backend.list(MemoryTier.WORKING)[0].metadata == {"source": "user"}
