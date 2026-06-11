"""Cost-ladder write path: L0 → L1 → KL gate → L2."""

from __future__ import annotations

from dataclasses import dataclass

from mnemo.embeddings.base import Embedder
from mnemo.extraction.l0_extractor import extract_l0
from mnemo.extraction.l1_extractor import extract_l1
from mnemo.extraction.l2_extractor import LLMExtractor
from mnemo.extraction.templates import TemplateLibrary
from mnemo.models import MemoryItem
from mnemo.policy import MemoryPolicy
from mnemo.scoring.kl_scorer import kl_surprise
from mnemo.types import WriteResult


@dataclass(frozen=True)
class IngestPipelineResult:
    """Outcome of running the full write ladder on one utterance."""

    semantic_writes: list[WriteResult]
    write_level: int | None
    kl_score: float | None
    total_cost_usd: float


def _belief_vectors(active_facts: list[MemoryItem], embedder: Embedder) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in active_facts:
        text = f"{item.metadata.get('entity')}:{item.metadata.get('predicate')}={item.value}"
        vectors.append(embedder.embed(str(text)))
    return vectors


def run_ingest_pipeline(
    text: str,
    *,
    embedder: Embedder,
    template_library: TemplateLibrary,
    policy: MemoryPolicy,
    active_semantic_facts: list[MemoryItem],
    store_l0,
    store_l1,
    store_l2,
    llm_extractor: LLMExtractor | None = None,
) -> IngestPipelineResult:
    """Execute L0, L1, KL-gated L2. Callbacks persist facts and return WriteResults."""
    total_cost = 0.0

    l0_facts = extract_l0(text)
    if l0_facts:
        writes = [store_l0(fact) for fact in l0_facts]
        cost = sum(w.cost_usd for w in writes)
        return IngestPipelineResult(writes, 0, None, cost)

    l1_batch = extract_l1(text, embedder, template_library, policy=policy)
    total_cost += l1_batch.cost_usd
    if l1_batch.facts:
        writes = [store_l1(fact, l1_batch.cost_usd / len(l1_batch.facts)) for fact in l1_batch.facts]
        return IngestPipelineResult(writes, 1, None, total_cost)

    utterance_vec = embedder.embed(text)
    beliefs = _belief_vectors(active_semantic_facts, embedder)
    kl = kl_surprise(utterance_vec, beliefs)

    if kl < policy.kl_threshold:
        return IngestPipelineResult([], None, kl, total_cost)

    if llm_extractor is None:
        return IngestPipelineResult([], None, kl, total_cost)

    l2_batch = llm_extractor.extract_facts(text)
    total_cost += l2_batch.cost_usd
    if not l2_batch.facts:
        return IngestPipelineResult([], None, kl, total_cost)

    share = l2_batch.cost_usd / len(l2_batch.facts)
    writes = [store_l2(fact, share) for fact in l2_batch.facts]
    return IngestPipelineResult(writes, 2, kl, total_cost)
