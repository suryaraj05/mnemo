"""Level 0 write path: deterministic regex extraction (zero API cost)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from mnemo.extraction.models import ExtractedFact

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_NAME_MY = re.compile(
    r"(?i)\b(?:my name is|call me)\s+"
    r"([A-Za-z][a-zA-Z]+(?:\s+[A-Za-z][a-zA-Z]+)*?)(?:\s+and\b|\.|,|$)",
)
_NAME_I_AM = re.compile(
    r"\b(?:I'm|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
)
_LOCATION = re.compile(
    r"(?i)\b(?:i live in|i(?:'m| am) from|i moved to)\s+"
    r"([A-Za-z][A-Za-z\s,]+?)(?:\s+last\b|\s+this\b|\.|,|$|\s+and\b)",
)

_L0_CONFIDENCE = 0.9


def extract_l0(
    text: str,
    *,
    default_entity: str = "user",
    now: datetime | None = None,
) -> list[ExtractedFact]:
    """Extract facts from ``text`` using compiled regex patterns.

    Returns an empty list when nothing matches. Same-turn duplicates for the
    same predicate are deduplicated (last match wins).
    """
    when = now or datetime.now(timezone.utc)
    by_predicate: dict[str, ExtractedFact] = {}

    def _add(predicate: str, value: str, confidence: float = _L0_CONFIDENCE) -> None:
        cleaned = value.strip().rstrip(".,")
        if not cleaned:
            return
        by_predicate[predicate] = ExtractedFact(
            entity=default_entity,
            predicate=predicate,
            value=cleaned,
            confidence=confidence,
            write_level=0,
            valid_from=when,
        )

    for match in _EMAIL.finditer(text):
        _add("email", match.group(0))

    for match in _ISO_DATE.finditer(text):
        _add("date", match.group(1))

    for pattern in (_NAME_MY, _NAME_I_AM):
        for match in pattern.finditer(text):
            _add("name", match.group(1))

    for match in _LOCATION.finditer(text):
        _add("location", match.group(1))

    return list(by_predicate.values())
