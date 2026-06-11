"""Level 1 write path: template embedding match + regex value capture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from mnemo.embeddings.base import Embedder
from mnemo.embeddings.similarity import cosine_similarity
from mnemo.extraction.models import ExtractedFact
from mnemo.extraction.templates import FactTemplate, TemplateLibrary
from mnemo.policy import MemoryPolicy

_L1_CONFIDENCE = 0.75


@dataclass(frozen=True)
class L1ExtractionResult:
    """Outcome of one L1 extraction pass."""

    facts: list[ExtractedFact]
    cost_usd: float
    embed_calls: int


def _extract_value(text: str, template: FactTemplate) -> str | None:
    match = template.value_regex.search(text)
    if not match:
        return None
    return match.group(1).strip().rstrip(".,")


def _best_score(
    query_vector: list[float],
    template: FactTemplate,
    library: TemplateLibrary,
    embedder: Embedder,
) -> float:
    best = -1.0
    for utterance in template.utterances:
        vector = library.vector_for(utterance)
        if vector is None:
            vector = embedder.embed(utterance)
            library.cache_vector(utterance, vector)
        score = cosine_similarity(query_vector, vector)
        if score > best:
            best = score
    return best


def extract_l1(
    text: str,
    embedder: Embedder,
    library: TemplateLibrary,
    *,
    policy: MemoryPolicy | None = None,
    threshold: float | None = None,
    cost_per_embed: float | None = None,
    default_entity: str = "user",
    now: datetime | None = None,
) -> L1ExtractionResult:
    """Match ``text`` against template utterances; extract values above threshold."""
    cfg = policy or MemoryPolicy()
    cutoff = threshold if threshold is not None else cfg.l1_cosine_threshold
    unit_cost = cost_per_embed if cost_per_embed is not None else cfg.l1_embed_cost_usd
    when = now or datetime.now(timezone.utc)

    if not library.is_warmed():
        library.warm_cache(embedder)

    query_vector = embedder.embed(text)
    embed_calls = 1
    cost_usd = unit_cost * embed_calls

    by_predicate: dict[str, ExtractedFact] = {}
    for template in library.templates:
        if template.entity != default_entity:
            continue
        score = _best_score(query_vector, template, library, embedder)
        if score < cutoff:
            continue
        value = _extract_value(text, template)
        if not value:
            continue
        confidence = min(_L1_CONFIDENCE, max(0.0, score))
        by_predicate[template.predicate] = ExtractedFact(
            entity=template.entity,
            predicate=template.predicate,
            value=value,
            confidence=confidence,
            write_level=1,
            valid_from=when,
        )

    return L1ExtractionResult(
        facts=list(by_predicate.values()),
        cost_usd=cost_usd,
        embed_calls=embed_calls,
    )
