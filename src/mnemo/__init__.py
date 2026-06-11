"""Mnemo: a 4-tier, backend-agnostic, cost-aware memory library for LLM agents.

Phases 0–7: backends, tiers, embeddings, L0 extraction.
"""

from mnemo.backends import (
    ExactVectorBackend,
    InMemoryBackend,
    MemoryBackend,
    SQLiteBackend,
    VectorBackend,
)
from mnemo.embeddings import Embedder, HashEmbedder, cosine_similarity
from mnemo.models import ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.extraction import ExtractedFact, extract_l0
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
    "extract_l0",
    "ForgetScope",
    "MemoryItem",
    "MemoryTier",
    "MemoryPolicy",
    "load_policy",
    "DeleteResult",
    "ReadResult",
    "WriteResult",
]
