"""Mnemo: a 4-tier, backend-agnostic, cost-aware memory library for LLM agents.

Phases 0–9: backends, tiers, embeddings, L0/L1 extraction, temporal decay.
"""

from mnemo.backends import (
    ExactVectorBackend,
    InMemoryBackend,
    MemoryBackend,
    SQLiteBackend,
    VectorBackend,
)
from mnemo.embeddings import Embedder, HashEmbedder, cosine_similarity
from mnemo.decay import decay_weight, weight_exponential, weight_power_law
from mnemo.models import DecayMode, ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.extraction import (
    ExtractedFact,
    FactTemplate,
    L1ExtractionResult,
    TemplateLibrary,
    extract_l0,
    extract_l1,
    load_template_library,
)
from mnemo.tiers import EpisodicMemory, SemanticMemory, WorkingMemory
from mnemo.types import DeleteResult, ReadResult, WriteResult

__version__ = "0.1.0a0"

__all__ = [
    "__version__",
    "ExactVectorBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "MemoryBackend",
    "VectorBackend",
    "Embedder",
    "HashEmbedder",
    "cosine_similarity",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ExtractedFact",
    "FactTemplate",
    "L1ExtractionResult",
    "TemplateLibrary",
    "extract_l0",
    "extract_l1",
    "load_template_library",
    "DecayMode",
    "decay_weight",
    "weight_exponential",
    "weight_power_law",
    "ForgetScope",
    "MemoryItem",
    "MemoryTier",
    "MemoryPolicy",
    "load_policy",
    "DeleteResult",
    "ReadResult",
    "WriteResult",
]
