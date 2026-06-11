# Mnemo Build Log

## Phase 0 - Foundation + contracts (2026-06-11)

### Shipped
- src-layout package, hatchling build, Python >=3.10, pydantic>=2, pytest dev extra
- `models.py`: `MemoryTier` (working/episodic/semantic/procedural), `ForgetScope`
  (item/tier/all/entity - entity reserved for GDPR), `MemoryItem(key, value, metadata)`
- `types.py`: `WriteResult` / `ReadResult` / `DeleteResult`, all carrying
  `cost_usd`, `latency_ms`, tz-aware `timestamp`; `WriteResult.write_level` encodes
  the L0/L1/L2 cost ladder for future audit logging
- `policy.py`: empty `MemoryPolicy` (extra="forbid") with the planned-field roadmap in
  the docstring; `load_policy` raises `NotImplementedError`
- `backends/base.py`: `MemoryBackend` ABC - exact frozen signatures for
  `write`, `read`, `delete`, `list`
- ADR-001 (why 4 tiers, backend abstraction, cost ladder), README, CI test stage
- 25 contract test cases (incl. parametrized) in `tests/test_models.py`

### Deliberately NOT built (per scope)
InMemoryBackend, SQLiteBackend, tiers/, embeddings, remember/recall/forget,
benchmarks, MCP, CLI.

### Notes for next session (Phase 1)
- Implement `InMemoryBackend` against the ABC; property-style tests for
  write/read/delete/list round-trips and filter semantics
- Implement WORKING-tier eviction: min-heap on `(importance, arrival_time, key)`,
  bounded by `MemoryPolicy.max_working_size` (add the field, un-comment roadmap)
- Decide `read()` relevance semantics for non-vector backends (substring + recency?)
- Keep backends synchronous; async stays in the future public API layer

### Open questions
- Should `delete` be strictly idempotent (current docstring says yes) or report misses?
  `DeleteResult.deleted_count` supports either - decide in Phase 1.
- `metadata` conventions per tier (importance, event_time vs ingest_time) need a schema
  note before EPISODIC lands. -> Resolved in Phase 1: docs/metadata-schema.md
- YAML dependency choice for `load_policy` (PyYAML vs ruamel) deferred.

## Phase 1 - InMemoryBackend + pinned semantics (2026-06-11)

### Shipped
- ADR-002: pinned backend semantics - case-sensitive substring read on key or
  str(value), insertion-order results sliced to top_k (top_k <= 0 -> []), idempotent
  silent-no-op delete, upsert write with metadata None -> {} and defensive copy
  (overwrite keeps original insertion position), list with metadata.get(k) == v
  exact-match filters (None == {} == all)
- `src/mnemo/backends/memory.py`: `InMemoryBackend` (dict[MemoryTier, dict[str, MemoryItem]]),
  exported from `mnemo.backends` and `mnemo`
- `docs/metadata-schema.md`: reserved metadata key conventions per tier (not enforced)
- 17 new contract tests in `tests/test_in_memory_backend.py` (42 total)
- README: Phase 1 status + quick example

### Deliberately NOT built (per scope)
WorkingMemory eviction heap (Phase 2), EpisodicMemory controller (Phase 3),
SQLiteBackend (Phase 4), MemoryPolicy changes.

### Notes for next session (Phase 2)
- WorkingMemory controller: bounded tier, min-heap on (importance, t, key);
  add `max_working_size` to MemoryPolicy (un-comment from roadmap, validate > 0)
- Heap is controller state layered over any MemoryBackend - do not push eviction
  logic into backends
- Use metadata keys `importance` and `t` per docs/metadata-schema.md

### Open questions
- Case-insensitive read: opt-in constructor flag later? ADR-002 default stays case-sensitive.
- `str(value)` matching stringifies large payloads on every read - fine for the
  reference backend; revisit if used beyond tests/notebooks.
- Thread safety is explicitly out of scope for InMemoryBackend - decide where
  concurrency guarantees live (backend vs orchestration) before SQLite (Phase 4).

## Phase 2 - WorkingMemory controller (2026-06-11)

### Shipped
- `MemoryPolicy.max_working_size` (default 10, ge=1)
- `src/mnemo/tiers/working.py`: min-heap eviction, Policy B, lazy heap staleness
- ADR-003, 13 tests in `tests/test_working_memory.py`

## Phase 3 - EpisodicMemory controller (2026-06-11)

### Shipped
- `src/mnemo/tiers/episodic.py`: record, recall_recent, get_timeline, retire
- ADR-004, 6 tests in `tests/test_episodic_memory.py`

## Phase 4 - SQLiteBackend (2026-06-11)

### Shipped
- `src/mnemo/backends/sqlite.py`: JSON value/metadata, INSERT OR REPLACE
- Episodic persistence integration test
- 5 tests in `tests/test_sqlite_backend.py`

## Phase 5 - Embedding layer (2026-06-11)

### Shipped
- `Embedder` ABC, `cosine_similarity`, `normalize_l2`, zero-norm guard
- `HashEmbedder` for CI (deterministic, no model download)
- `SentenceTransformerEmbedder` optional via `pip install "mnemo[local-embeddings]"`
- ADR-005, 13 tests in `tests/test_embeddings.py` (81 total)

### Notes for next session (Phase 6)
- Pgvector backend + vector column on episodic/semantic rows
- Semantic read path using cosine over stored embeddings
