"""Phases 15-16: Mnemo remember / recall / forget."""

from __future__ import annotations

from mnemo import ForgetScope, InMemoryBackend, MemoryPolicy, MemoryTier, Mnemo


def test_remember_l0_and_recall() -> None:
    mnemo = Mnemo(InMemoryBackend(), MemoryPolicy())
    _, pipeline = mnemo.remember("My name is Priya")
    assert pipeline.write_level == 0
    result = mnemo.recall("Priya", top_k=5)
    assert len(result.items) >= 1


def test_forget_entity_removes_semantic_facts() -> None:
    mnemo = Mnemo(InMemoryBackend(), MemoryPolicy())
    mnemo.remember("My name is Sam")
    assert mnemo.semantic.get_fact("user", "name") is not None
    deleted = mnemo.forget(ForgetScope.ENTITY, entity="user")
    assert deleted.deleted_count >= 1
    assert mnemo.semantic.get_fact("user", "name") is None


def test_forget_item() -> None:
    backend = InMemoryBackend()
    mnemo = Mnemo(backend, MemoryPolicy())
    evt_key, _ = mnemo.remember("hello")
    result = mnemo.forget(ForgetScope.ITEM, tier=MemoryTier.EPISODIC, key=evt_key)
    assert result.deleted_count == 1
