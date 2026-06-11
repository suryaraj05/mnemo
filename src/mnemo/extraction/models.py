"""Structured facts produced by extractors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExtractedFact:
    """One entity–predicate–value fact from an extractor."""

    entity: str
    predicate: str
    value: str
    confidence: float
    write_level: int = 0
    valid_from: datetime | None = None
