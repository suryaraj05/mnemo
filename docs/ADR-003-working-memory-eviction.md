# ADR-003: Working memory eviction controller

## Status

Accepted (Phase 2)

## Context

WORKING tier must stay bounded. Eviction uses importance with FIFO tie-breaking.

## Decision

1. **Eviction lives in `WorkingMemory`**, not in backends (ADR-002 backends stay dumb).
2. **Min-heap** on `(importance, t, key)`; evict lexicographic minimum.
3. **Policy B**: when full, reject new item unless it beats heap minimum.
4. **Heap staleness**: `_entries` dict maps `key → (importance, t)` as canonical identity;
   stale heap tuples are discarded lazily on peek/pop when identity mismatches.
5. **Same-key overwrite**: updates backend row and pushes new heap tuple; does not consume
   an extra capacity slot.
6. **Rejected adds still increment `t`** — `t` is attempt order, not stored-only order.
7. **Persistence**: `_heap` and `_time` are ephemeral until a future controller-rebuild story
   lands with SQLite (Phase 4 stores rows only).

## Consequences

- Deletes bypassing the controller can desync heap; only `WorkingMemory.add/evict_one` should
  write WORKING tier in production paths.
- Single-threaded assumption; locking belongs in orchestration layer later.
