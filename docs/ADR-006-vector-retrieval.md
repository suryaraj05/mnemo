# ADR-006: Vector retrieval and pgvector

## Status

Accepted (Phase 6)

## Context

Episodic recall needs semantic similarity (Phase 5 embeddings) with a production ANN path
and a CI-exact reference implementation.

## Decision

1. **`VectorBackend`** extends `MemoryBackend` with `search_by_vector(tier, query_vector, top_k)`.
2. **`ExactVectorBackend`**: brute-force cosine over `metadata["embedding"]` — CI default
   (MemPalace `sqlite_exact` pattern).
3. **`PgvectorBackend`**: PostgreSQL + `vector` extension + **HNSW** index with
   `vector_cosine_ops`; query uses `<=>` cosine distance operator.
4. Embeddings stored under metadata key **`embedding`** on write; Pgvector extracts to
   `vector(d)` column on persist.
5. **`EpisodicMemory.recall_semantic`**: embed query, search, filter `txn_to is null`.
6. **Remote DSN warning** when host is not localhost (explicit opt-in, MemPalace pattern).
7. Pgvector tests **skipped in CI** unless `MNEMO_PGVECTOR_DSN` is set.

## Consequences

- HNSW is approximate; ExactVectorBackend validates correctness in CI.
- Fixed `dimension` at backend init; mismatch on write is a DB error.
- Namespace prefix isolates tenants via `MNEMO_PGVECTOR_NAMESPACE`.
