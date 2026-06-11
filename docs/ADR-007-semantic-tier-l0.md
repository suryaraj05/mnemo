# ADR-007: Semantic tier + L0 regex extraction

## Status

Accepted (Phase 7)

## Context

Semantic tier holds extracted facts (entity–predicate–value), distinct from episodic
verbatim events. The write-path cost ladder starts at Level 0: free deterministic regex.

## Decision

1. **`SemanticMemory`** stores facts on `MemoryTier.SEMANTIC` with metadata:
   `entity`, `predicate`, `valid_from`, `valid_to`, `txn_from`, `txn_to`,
   `confidence`, `source`, optional `write_level`.
2. **Logical identity** is `(entity, predicate)`; physical keys are `fact_N`.
3. **Conflict resolution**: new value for same `(entity, predicate)` closes the active
   row (`txn_to` + `valid_to` set) and inserts a new row — history preserved.
4. **Idempotent store**: same active value → no new row (`WriteResult.created=False`).
5. **`extract_l0`**: regex for email, ISO date, name, location; `write_level=0`,
   `source="l0:regex"`, confidence `0.9`.
6. **Name guard**: `I'm/I am` requires capitalized token to avoid adjective false
   positives (`I am happy` does not extract).

## Consequences

- Episodic stays verbatim; semantic holds normalized facts for recall and later KL gating.
- L1 template matching (Phase 8) will call `store_fact` with `write_level=1`.
- `get_history` exposes full bi-temporal audit for corrections (e.g. Alex → Alexander).
