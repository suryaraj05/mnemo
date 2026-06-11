"""Phase 0 contract tests: models, result types, policy stub, backend ABC."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mnemo import __version__
from mnemo.backends import MemoryBackend
from mnemo.models import ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.types import DeleteResult, ReadResult, WriteResult


class TestEnums:
    def test_memory_tier_values(self) -> None:
        assert [t.value for t in MemoryTier] == [
            "working",
            "episodic",
            "semantic",
            "procedural",
        ]

    def test_memory_tier_is_str(self) -> None:
        assert isinstance(MemoryTier.WORKING, str)
        assert MemoryTier("episodic") is MemoryTier.EPISODIC

    def test_forget_scope_values(self) -> None:
        assert {s.value for s in ForgetScope} == {"item", "tier", "all", "entity"}

    def test_forget_scope_is_str(self) -> None:
        assert isinstance(ForgetScope.ENTITY, str)


class TestMemoryItem:
    def test_requires_key(self) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(value="orphan")  # type: ignore[call-arg]

    def test_requires_value(self) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(key="k")  # type: ignore[call-arg]

    def test_metadata_defaults_to_empty_dict(self) -> None:
        item = MemoryItem(key="k", value=1)
        assert item.metadata == {}

    def test_metadata_default_not_shared_between_instances(self) -> None:
        a = MemoryItem(key="a", value=1)
        b = MemoryItem(key="b", value=2)
        a.metadata["importance"] = 0.9
        assert b.metadata == {}

    @pytest.mark.parametrize(
        "payload",
        ["text", 42, 3.14, None, {"nested": [1, 2]}, ["a", "b"]],
    )
    def test_value_accepts_arbitrary_payloads(self, payload: Any) -> None:
        assert MemoryItem(key="k", value=payload).value == payload

    def test_key_must_be_string(self) -> None:
        with pytest.raises(ValidationError):
            MemoryItem(key=123, value="x")  # type: ignore[arg-type]


class TestResultTypes:
    def test_write_result_defaults(self) -> None:
        r = WriteResult(tier=MemoryTier.EPISODIC, key="evt-1")
        assert r.created is True
        assert r.write_level is None
        assert r.cost_usd == 0.0
        assert r.timestamp.tzinfo is not None

    def test_read_result_holds_items(self) -> None:
        items = [MemoryItem(key="k", value="v")]
        r = ReadResult(query="q", items=items, tiers_searched=[MemoryTier.SEMANTIC], top_k=5)
        assert r.items == items
        assert r.tiers_searched == [MemoryTier.SEMANTIC]

    def test_delete_result_defaults(self) -> None:
        r = DeleteResult()
        assert r.scope is ForgetScope.ITEM
        assert r.deleted_count == 0
        assert r.tier is None and r.key is None


class TestPolicy:
    def test_policy_default_max_working_size(self) -> None:
        assert MemoryPolicy().max_working_size == 10

    def test_policy_accepts_max_working_size(self) -> None:
        assert MemoryPolicy(max_working_size=5).max_working_size == 5

    def test_policy_rejects_invalid_max_working_size(self) -> None:
        with pytest.raises(ValidationError):
            MemoryPolicy(max_working_size=0)

    def test_policy_accepts_kl_threshold(self) -> None:
        assert MemoryPolicy(kl_threshold=0.5).kl_threshold == 0.5

    def test_policy_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            MemoryPolicy(unknown_field=1)  # type: ignore[call-arg]

    def test_load_policy_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_policy("policy-does-not-exist.yaml")


class TestMemoryBackendABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            MemoryBackend()  # type: ignore[abstract]

    def test_minimal_subclass_satisfies_contract(self) -> None:
        class StubBackend(MemoryBackend):
            def write(
                self,
                tier: MemoryTier,
                key: str,
                value: Any,
                metadata: dict[str, Any] | None = None,
            ) -> None:
                return None

            def read(self, tier: MemoryTier, query: str, top_k: int) -> list[MemoryItem]:
                return []

            def delete(self, tier: MemoryTier, key: str) -> None:
                return None

            def list(
                self,
                tier: MemoryTier,
                filters: dict[str, Any] | None = None,
            ) -> list[MemoryItem]:
                return []

        backend = StubBackend()
        assert backend.read(MemoryTier.WORKING, "query", 5) == []
        assert backend.list(MemoryTier.PROCEDURAL) == []

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        class Partial(MemoryBackend):
            def write(
                self,
                tier: MemoryTier,
                key: str,
                value: Any,
                metadata: dict[str, Any] | None = None,
            ) -> None:
                return None

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str) and __version__
