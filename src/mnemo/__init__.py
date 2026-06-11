"""Mnemo: a 4-tier, backend-agnostic, cost-aware memory library for LLM agents.

Phase 0 exposes only the core contracts: models, result types, the policy stub,
and the :class:`~mnemo.backends.base.MemoryBackend` ABC.
"""

from mnemo.models import ForgetScope, MemoryItem, MemoryTier
from mnemo.policy import MemoryPolicy, load_policy
from mnemo.types import DeleteResult, ReadResult, WriteResult

__version__ = "0.1.0a0"

__all__ = [
    "__version__",
    "ForgetScope",
    "MemoryItem",
    "MemoryTier",
    "MemoryPolicy",
    "load_policy",
    "DeleteResult",
    "ReadResult",
    "WriteResult",
]
