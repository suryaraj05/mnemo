"""Provider-agnostic embedding contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Maps text to dense vectors for semantic similarity retrieval."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector length ``d`` for all outputs from this provider."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single string."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings.

        Default loops ``embed``; providers should override for batching efficiency.
        """
        return [self.embed(text) for text in texts]
