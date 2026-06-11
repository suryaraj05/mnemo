"""Level 2 write path: provider-agnostic LLM fact extraction (Phase 12)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from mnemo.extraction.models import ExtractedFact

_L2_CONFIDENCE = 0.65


@dataclass(frozen=True)
class L2ExtractionResult:
    facts: list[ExtractedFact]
    cost_usd: float


class LLMExtractor(ABC):
    """Extract structured facts from free text via an LLM provider."""

    @abstractmethod
    def extract_facts(self, text: str) -> L2ExtractionResult:
        """Return facts and marginal USD cost for this call."""


class MockLLMExtractor(LLMExtractor):
    """Deterministic extractor for CI — no network calls."""

    def __init__(
        self,
        mapping: dict[str, list[tuple[str, str, str]]] | None = None,
        *,
        default_cost_usd: float = 0.01,
    ) -> None:
        self._mapping = mapping or {}
        self._default_cost = default_cost_usd

    def extract_facts(self, text: str) -> L2ExtractionResult:
        when = datetime.now(timezone.utc)
        rows = self._mapping.get(text, [])
        facts = [
            ExtractedFact(
                entity=entity,
                predicate=predicate,
                value=value,
                confidence=_L2_CONFIDENCE,
                write_level=2,
                valid_from=when,
            )
            for entity, predicate, value in rows
        ]
        return L2ExtractionResult(facts=facts, cost_usd=self._default_cost if rows else 0.0)
