# ADR-002: InMemoryBackend semantics (reference for non-vector backends)

- Status: Accepted
- Date: 2026-06-11
- Phase: 1
- Supersedes: nothing; refines the `MemoryBackend` contract from ADR-001

## Context

The `MemoryBackend` ABC (Phase 0) froze signatures but left retrieval and edge-case
semantics open. Before the first concrete backend ships, those semantics must be pinned
so every future backend (SQLite, vector stores) and every tier controller built on top
can rely on identical behavior. `InMemoryBackend` is the executable specification.

## Decisions

### read(tier, query, top_k)

- **Matching**: substring match of `query` against `item.key` OR `str(item.value)`.
  No tokenization, no ranking model.
- **Case sensitivity**: case-sensitive. A `casefold` option may be added later as an
  opt-in constructor flag; default behavior will not change.
- **Ordering**: insertion order among matches (stable), then sliced to `[:top_k]`.
  There is no relevance score in non-vector backends; first-written wins ties by design.
- **Upsert position**: overwriting an existing key keeps its original insertion
  position (CPython dict semantics, documented as contract, not accident).
- **Edge cases**: missing/empty tier bucket returns `[]` (never an error).
  `top_k <= 0` returns `[]`.

### delete(tier, key)

- **Idempotent**, per the Phase 0 ABC docstring: deleting from a missing tier or a
  missing key is a **silent no-op** — no exception, no return value (method returns
  `None`). Callers that need an audit trail use `DeleteResult.deleted_count` at the
  orchestration layer (later phases), not the backend.
- The Phase 0 ABC docstring already states this; no docstring change required.

### write(tier, key, value, metadata)

- **Upsert**: writing an existing `(tier, key)` overwrites value and metadata entirely
  (no merge).
- **`metadata=None` is stored as `{}`**; backends must never store `None` metadata.
- **Defensive copy**: the backend stores a shallow copy of `metadata` so later mutation
  of the caller's dict cannot silently alter stored state.

### list(tier, filters)

- `filters=None` and `filters={}` are equivalent: return **all** items in the tier.
- **Match rule**: an item matches iff every `(k, v)` pair in `filters` satisfies
  `item.metadata.get(k) == v`. A missing metadata key yields `None`, which only matches
  an explicit `None` filter value — so items lacking the key are excluded for any
  non-None filter value.
- Missing tier bucket returns `[]`.
- Ordering: insertion order (same rule as `read`).

## Consequences

- Tier controllers (Phase 2+) can rely on deterministic ordering for eviction and
  pagination without backend-specific branches.
- Substring-on-`str(value)` is intentionally naive; large structured payloads will
  stringify on every read. Acceptable for the reference backend; indexed backends
  override the mechanism but must preserve the observable contract (stable order,
  `[]` on miss, idempotent delete).
- `InMemoryBackend` is not thread-safe and not persistent; it is the spec and the
  test double, not a production store.
