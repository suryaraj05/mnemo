"""Mnemo: a 4-tier, backend-agnostic, cost-aware memory library for LLM agents."""

from mnemo.api import Mnemo
from mnemo.backends import (
    CompositeBackend,
    ExactVectorBackend,
    InMemoryBackend,
    MemoryBackend,
    SQLiteBackend,
    VectorBackend,
)
from mnemo.decay import decay_weight, weight_exponential, weight_power_law
from mnemo.embeddings import Embedder, HashEmbedder, cosine_similarity
from mnemo.extraction import (
    ExtractedFact,
    FactTemplate,
    L1ExtractionResult,
    L2ExtractionResult,
    LLMExtractor,
    MockLLMExtractor,
    TemplateLibrary,
    extract_l0,
    extract_l1,
    load_template_library,
)
from mnemo.models import DecayMode, ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.scoring import kl_divergence, kl_surprise
from mnemo.tiers import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from mnemo.types import DeleteResult, ReadResult, WriteResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Mnemo",
    "CompositeBackend",
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
    "ProceduralMemory",
    "ExtractedFact",
    "FactTemplate",
    "L1ExtractionResult",
    "L2ExtractionResult",
    "LLMExtractor",
    "MockLLMExtractor",
    "TemplateLibrary",
    "extract_l0",
    "extract_l1",
    "load_template_library",
    "DecayMode",
    "decay_weight",
    "weight_exponential",
    "weight_power_law",
    "kl_divergence",
    "kl_surprise",
    "ForgetScope",
    "MemoryItem",
    "MemoryTier",
    "MemoryPolicy",
    "load_policy",
    "DeleteResult",
    "ReadResult",
    "WriteResult",
]
