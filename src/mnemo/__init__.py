"""Mnemo: a 4-tier, backend-agnostic, cost-aware memory library for LLM agents.

Phases 0–4: contracts, InMemory + SQLite backends, Working + Episodic tiers.
"""

from mnemo.backends import InMemoryBackend, MemoryBackend, SQLiteBackend
from mnemo.models import ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.tiers import EpisodicMemory, WorkingMemory
from mnemo.types import DeleteResult, ReadResult, WriteResult

__version__ = "0.1.0a0"

__all__ = [
    "__version__",
    "InMemoryBackend",
    "SQLiteBackend",
    "MemoryBackend",
    "WorkingMemory",
    "EpisodicMemory",
    "ForgetScope",
    "MemoryItem",
    "MemoryTier",
    "MemoryPolicy",
    "load_policy",
    "DeleteResult",
    "ReadResult",
    "WriteResult",
]
