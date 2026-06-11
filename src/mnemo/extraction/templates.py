"""L1 template library: example utterances per fact slot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mnemo.embeddings.base import Embedder


class FactTemplate(BaseModel):
    """One semantic slot with embedding exemplars and a value regex."""

    entity: str
    predicate: str
    utterances: list[str] = Field(min_length=1)
    value_pattern: str

    @field_validator("value_pattern")
    @classmethod
    def _compile_check(cls, pattern: str) -> str:
        re.compile(pattern)
        return pattern

    @property
    def value_regex(self) -> re.Pattern[str]:
        return re.compile(self.value_pattern)


@dataclass
class TemplateLibrary:
    """Collection of fact templates with optional pre-computed utterance vectors."""

    templates: list[FactTemplate]
    _vectors: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def warm_cache(self, embedder: Embedder) -> int:
        """Embed all template utterances in one batch. Returns utterance count."""
        missing = [u for t in self.templates for u in t.utterances if u not in self._vectors]
        if not missing:
            return 0
        vectors = embedder.embed_batch(missing)
        for utterance, vector in zip(missing, vectors):
            self._vectors[utterance] = vector
        return len(missing)

    def is_warmed(self) -> bool:
        total = sum(len(t.utterances) for t in self.templates)
        return len(self._vectors) >= total

    def vector_for(self, utterance: str) -> list[float] | None:
        return self._vectors.get(utterance)

    def cache_vector(self, utterance: str, vector: list[float]) -> None:
        self._vectors[utterance] = vector


def _parse_library_payload(data: dict[str, Any]) -> TemplateLibrary:
    templates = [FactTemplate.model_validate(entry) for entry in data["templates"]]
    return TemplateLibrary(templates=templates)


def load_template_library(path: str | Path | None = None) -> TemplateLibrary:
    """Load templates from JSON file or the bundled default library."""
    if path is None:
        raw = resources.files("mnemo.extraction.data").joinpath("l1_templates.json").read_text(
            encoding="utf-8"
        )
    else:
        raw = Path(path).read_text(encoding="utf-8")
    return _parse_library_payload(json.loads(raw))
