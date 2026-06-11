"""Unified Mnemo facade: remember, recall, forget (Phases 15–16)."""

from __future__ import annotations

import time
from typing import Any

from mnemo.api.forget import forget as forget_items
from mnemo.api.recall import composite_recall
from mnemo.backends.base import MemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.embeddings.hash_embedder import HashEmbedder
from mnemo.extraction.l2_extractor import LLMExtractor
from mnemo.extraction.models import ExtractedFact
from mnemo.extraction.pipeline import IngestPipelineResult, run_ingest_pipeline
from mnemo.extraction.templates import TemplateLibrary, load_template_library
from mnemo.models import ForgetScope, MemoryTier
from mnemo.policy import MemoryPolicy
from mnemo.tiers.episodic import EpisodicMemory
from mnemo.tiers.procedural import ProceduralMemory
from mnemo.tiers.semantic import SemanticMemory
from mnemo.tiers.working import WorkingMemory
from mnemo.types import DeleteResult, ReadResult, WriteResult


class Mnemo:
    """Backend-agnostic memory orchestrator for all four tiers."""

    def __init__(
        self,
        backend: MemoryBackend,
        policy: MemoryPolicy | None = None,
        *,
        embedder: Embedder | None = None,
        template_library: TemplateLibrary | None = None,
        llm_extractor: LLMExtractor | None = None,
    ) -> None:
        self._backend = backend
        self._policy = policy or MemoryPolicy()
        self._embedder = embedder or HashEmbedder()
        self._templates = template_library or load_template_library()
        self._llm = llm_extractor
        self._working = WorkingMemory.from_policy(backend, self._policy.max_working_size)
        self._episodic = EpisodicMemory(backend)
        self._semantic = SemanticMemory(backend)
        self._procedural = ProceduralMemory(backend, self._policy.procedural_min_successes)

    @property
    def policy(self) -> MemoryPolicy:
        return self._policy

    @property
    def working(self) -> WorkingMemory:
        return self._working

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def procedural(self) -> ProceduralMemory:
        return self._procedural

    def remember(
        self,
        text: str,
        *,
        source: str = "user",
        importance: float = 0.5,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, IngestPipelineResult]:
        """Store a turn: episodic verbatim + working slot + semantic ladder."""
        start = time.perf_counter()
        evt_key = self._episodic.record(
            text,
            source,
            scope=scope,
            metadata=metadata,
            embedder=self._embedder,
        )
        self._working.add(evt_key, text, importance=importance)

        def store_l0(fact: ExtractedFact) -> WriteResult:
            return self._semantic._store_extracted(fact, "l0:regex")

        def store_l1(fact: ExtractedFact, cost: float) -> WriteResult:
            return self._semantic._store_extracted(fact, "l1:template", cost_usd=cost)

        def store_l2(fact: ExtractedFact, cost: float) -> WriteResult:
            return self._semantic._store_extracted(fact, "l2:llm", cost_usd=cost)

        pipeline = run_ingest_pipeline(
            text,
            embedder=self._embedder,
            template_library=self._templates,
            policy=self._policy,
            active_semantic_facts=self._semantic.recall_facts(),
            store_l0=store_l0,
            store_l1=store_l1,
            store_l2=store_l2,
            llm_extractor=self._llm,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        for write in pipeline.semantic_writes:
            write.latency_ms = latency_ms
        return evt_key, pipeline

    def recall(self, query: str, top_k: int = 10) -> ReadResult:
        """Tier-weighted hybrid recall across all tiers."""
        start = time.perf_counter()
        items, tiers = composite_recall(
            self._backend,
            query,
            top_k,
            self._policy,
            self._embedder,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ReadResult(
            query=query,
            items=items,
            tiers_searched=tiers,
            top_k=top_k,
            latency_ms=latency_ms,
        )

    def forget(
        self,
        scope: ForgetScope,
        *,
        tier: MemoryTier | None = None,
        key: str | None = None,
        entity: str | None = None,
    ) -> DeleteResult:
        """Delete across tiers per ``ForgetScope``."""
        return forget_items(
            self._backend,
            scope,
            tier=tier,
            key=key,
            entity=entity,
        )
