"""Storage backends for Mnemo.

Phase 0 ships only the abstract contract. Concrete backends (in-memory,
SQLite, vector stores) arrive in later phases.
"""

from mnemo.backends.base import MemoryBackend

__all__ = ["MemoryBackend"]
