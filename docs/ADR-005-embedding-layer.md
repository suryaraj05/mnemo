# ADR-005: Provider-agnostic embedding layer

## Status

Accepted (Phase 5)

## Context

Semantic retrieval (Phase 6+) needs dense vectors and cosine scoring without tying Mnemo
to one vendor API.

## Decision

1. **`Embedder` ABC** with `embed`, `embed_batch`, and `dimension` property.
2. **`cosine_similarity`** in pure Python; zero-norm vectors return `0.0` (no divide-by-zero).
3. **`HashEmbedder`** for CI and offline tests — deterministic, zero extra deps.
4. **`SentenceTransformerEmbedder`** behind optional extra `local-embeddings` for real local
   semantic vectors (MemPalace-style zero-API benchmark path).
5. Normalization policy: embedders may return unnormalized vectors; cosine handles norms.
   `normalize_l2` available when dot-product on unit vectors is desired.

## Consequences

- CI never downloads ML models; integration tests use `HashEmbedder`.
- Production agents install `mnemo[local-embeddings]` or plug OpenAI/Gemini embedders in Phase 6+.
