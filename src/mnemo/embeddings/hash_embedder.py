"""Deterministic hash embedder for tests and zero-dependency CI."""

from __future__ import annotations

import hashlib
import struct

from mnemo.embeddings.base import Embedder


class HashEmbedder(Embedder):
    """Pseudo-embeddings from SHA-256 — deterministic, no ML model.

    Not semantically meaningful; use only for unit tests and offline dev.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vec: list[float] = []
        seed = text.encode("utf-8")
        while len(vec) < self._dimension:
            digest = hashlib.sha256(seed).digest()
            seed = digest
            for i in range(0, len(digest) - 3, 4):
                (value,) = struct.unpack("!f", digest[i : i + 4])
                if value != value:  # skip NaN from arbitrary bytes
                    continue
                vec.append(value)
                if len(vec) >= self._dimension:
                    break
        return vec[: self._dimension]
