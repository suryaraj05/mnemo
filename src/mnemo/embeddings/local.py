"""Optional local sentence-transformer embedder (requires extra dependency)."""

from __future__ import annotations

from mnemo.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings via ``sentence-transformers`` (offline after model download)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires the local extra: "
                'pip install "mnemo[local-embeddings]"'
            ) from exc

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine dimension for model {model_name!r}")
        self._dimension = int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        matrix = self._model.encode(texts, normalize_embeddings=False)
        return [row.tolist() for row in matrix]
