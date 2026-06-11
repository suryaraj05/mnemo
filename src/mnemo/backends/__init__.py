"""Storage backends for Mnemo.

Phase 1 ships :class:`InMemoryBackend`, the executable specification of
ADR-002. Persistent backends (SQLite, vector stores) arrive in later phases.
"""

from mnemo.backends.base import MemoryBackend
from mnemo.backends.memory import InMemoryBackend

__all__ = ["InMemoryBackend", "MemoryBackend"]
