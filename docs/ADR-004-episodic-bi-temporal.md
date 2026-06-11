# ADR-004: Episodic verbatim + bi-temporal metadata

## Status

Accepted (Phase 3)

## Context

Episodic tier stores raw experiences for audit, correction replay, and later semantic extraction.

## Decision

1. **`value` is verbatim text** — never summarized at write.
2. **`event_time`**: when the event happened in the world (ISO 8601 string).
3. **`txn_from` / `txn_to`**: when Mnemo stored / retired the row (`txn_to=None` = active).
4. **`recall_recent`**: active rows only, sort by `event_time` descending (newest world event first).
5. **`get_timeline`**: active rows, `event_time` ascending (replay order).
6. **`retire`**: sets `txn_to`, overwrites row via backend upsert — no hard delete.

## Consequences

- Retrieval uses **world time**, not `txn_from`, so backfilled rows rank correctly.
- Semantic summaries belong in SEMANTIC tier (later phases), not episodic write path.
