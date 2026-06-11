"""Phase 8 tests: L1 template embedding match."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnemo.embeddings.base import Embedder
from mnemo.extraction.l1_extractor import extract_l1
from mnemo.extraction.templates import FactTemplate, TemplateLibrary, load_template_library
from mnemo.policy import MemoryPolicy


class _ClusterEmbedder(Embedder):
    """Maps text clusters to fixed unit vectors for deterministic L1 tests."""

    def __init__(self) -> None:
        self._dim = 3

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        low = text.lower()
        if any(
            token in low
            for token in ("work at", "employer", "joined", "started at", "acme", "stripe")
        ):
            return [1.0, 0.0, 0.0]
        if any(token in low for token in ("guitar", "hobby", "hiking", "painting")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
def employment_library() -> TemplateLibrary:
    return TemplateLibrary(
        templates=[
            FactTemplate(
                entity="user",
                predicate="employer",
                utterances=["I work at Acme Corp", "My employer is Google"],
                value_pattern=(
                    r"(?i)(?:work at|employer is|joined|started at)\s+"
                    r"([A-Za-z][A-Za-z0-9\s&.-]+?)(?:\s+last|\s+yesterday|\s+this|\.|,|$)"
                ),
            )
        ]
    )


def test_load_default_template_library() -> None:
    lib = load_template_library()
    assert len(lib.templates) >= 2
    assert any(t.predicate == "employer" for t in lib.templates)


def test_load_template_library_from_file(tmp_path: Path) -> None:
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "entity": "user",
                        "predicate": "pet",
                        "utterances": ["I have a cat"],
                        "value_pattern": r"(?i)have a (\w+)",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    lib = load_template_library(path)
    assert lib.templates[0].predicate == "pet"


def test_extract_l1_employer_match(employment_library: TemplateLibrary) -> None:
    emb = _ClusterEmbedder()
    result = extract_l1(
        "I started at Stripe yesterday",
        emb,
        employment_library,
        threshold=0.85,
    )
    assert len(result.facts) == 1
    assert result.facts[0].predicate == "employer"
    assert result.facts[0].value == "Stripe"
    assert result.facts[0].write_level == 1
    assert result.facts[0].confidence <= 0.75


def test_extract_l1_below_threshold_returns_empty(employment_library: TemplateLibrary) -> None:
    emb = _ClusterEmbedder()
    result = extract_l1(
        "The weather is nice today",
        emb,
        employment_library,
        threshold=0.85,
    )
    assert result.facts == []


def test_extract_l1_cost_accounting(employment_library: TemplateLibrary) -> None:
    emb = _ClusterEmbedder()
    policy = MemoryPolicy(l1_embed_cost_usd=0.002)
    result = extract_l1(
        "I work at Acme Corp",
        emb,
        employment_library,
        policy=policy,
    )
    assert result.embed_calls == 1
    assert result.cost_usd == pytest.approx(0.002)
