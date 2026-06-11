"""Phase 7 tests: SemanticMemory + bi-temporal conflict resolution."""

from __future__ import annotations

from mnemo.backends.memory import InMemoryBackend
from mnemo.models import MemoryTier
from mnemo.tiers.semantic import SemanticMemory


def test_store_and_get_fact() -> None:
    sm = SemanticMemory(InMemoryBackend())
    sm.store_fact("user", "email", "a@b.com", "user")
    fact = sm.get_fact("user", "email")
    assert fact is not None
    assert fact.value == "a@b.com"
    assert fact.metadata["txn_to"] is None


def test_correction_closes_old_row_bi_temporally() -> None:
    sm = SemanticMemory(InMemoryBackend())
    sm.store_fact("user", "name", "Alex", "l0:regex", write_level=0)
    sm.store_fact("user", "name", "Alexander", "user")

    active = sm.get_fact("user", "name")
    assert active is not None
    assert active.value == "Alexander"

    history = sm.get_history("user", "name")
    assert len(history) == 2
    assert history[0].value == "Alex"
    assert history[0].metadata["txn_to"] is not None
    assert history[0].metadata["valid_to"] is not None
    assert history[1].value == "Alexander"
    assert history[1].metadata["txn_to"] is None


def test_same_value_is_idempotent() -> None:
    sm = SemanticMemory(InMemoryBackend())
    r1 = sm.store_fact("user", "location", "Pune", "user")
    r2 = sm.store_fact("user", "location", "Pune", "user")
    assert r1.created is True
    assert r2.created is False
    assert len(sm.recall_facts(entity="user")) == 1


def test_ingest_l0_from_text() -> None:
    sm = SemanticMemory(InMemoryBackend())
    results = sm.ingest_l0("My name is Priya and I live in Mumbai.")
    assert len(results) >= 2
    assert all(r.tier == MemoryTier.SEMANTIC for r in results)
    assert all(r.write_level == 0 for r in results)
    assert sm.get_fact("user", "name") is not None
    assert sm.get_fact("user", "location") is not None


def test_recall_facts_filter_by_entity() -> None:
    sm = SemanticMemory(InMemoryBackend())
    sm.store_fact("user", "name", "Sam", "user")
    sm.store_fact("project", "name", "Mnemo", "user")
    user_facts = sm.recall_facts(entity="user")
    assert len(user_facts) == 1
    assert user_facts[0].value == "Sam"
