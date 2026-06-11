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
  note before EPISODIC lands.
- YAML dependency choice for `load_policy` (PyYAML vs ruamel) deferred.
