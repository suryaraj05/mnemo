# ADR-001: Four-tier memory architecture, backend abstraction, and cost ladder

- Status: Accepted
- Date: 2026-06-11
- Phase: 0

## Context

LLM agents need memory that is simultaneously (a) cheap enough to write on every turn,
(b) faithful enough to audit, and (c) structured enough to recall selectively. A single
flat store fails all three: vector-only stores lose verbatim fidelity, context-window
buffers forget everything, and LLM-summarized stores are expensive and lossy at write time.

## Decision

### 1. Four tiers, modeled after human memory taxonomy

- **WORKING** - a bounded, fast tier for the current task. Eviction is
  importance-weighted FIFO implemented as a min-heap keyed on
  `(importance, arrival_time, key)`: the least important, oldest item is evicted first,
  with `key` as a deterministic tiebreaker.
- **EPISODIC** - verbatim, timestamped events. We never summarize at write time, because
  summarization is lossy and irreversible; compression decisions belong to read/consolidation
  time. Items carry bi-temporal metadata (event time vs. ingestion time) so corrections can
  be recorded without rewriting history.
- **SEMANTIC** - extracted facts with validity windows (facts expire or get superseded) and
  confidence scores (extraction is probabilistic; confidence drives recall ranking and
  re-verification).
- **PROCEDURAL** - behavioral patterns ("when X, do Y") promoted only after N confirmed
  successes, preventing one-off coincidences from becoming habits.

Separate tiers let policy treat each differently: eviction applies to WORKING only,
GDPR-style entity erasure must span all tiers, and recall can weight tiers per query.

### 2. Backend abstraction

All persistence goes through a small `MemoryBackend` ABC (`write`, `read`, `delete`, `list`).
Rationale:

- Tier logic (eviction, promotion, validity) must not be coupled to a storage engine.
- Deployments range from ephemeral in-memory (tests, notebooks) to SQLite (local agents) to
  hosted vector stores (production). One contract, many engines.
- A narrow synchronous interface keeps backends trivial to implement; async orchestration,
  batching, and caching live above this layer in the public API.

### 3. Cost ladder for writes

Every write attempt escalates only as far as needed:

- **Level 0 (regex)**: deterministic patterns (emails, dates, IDs). Free.
- **Level 1 (template embedding match)**: cosine similarity against known extraction
  templates. Cheap, no generation.
- **Level 2 (LLM extraction)**: invoked only when the KL divergence between the observed
  turn distribution and what Levels 0-1 explain exceeds a policy threshold - i.e., only
  when the turn contains genuinely novel information worth paying for.

This makes cost a first-class, policy-controlled budget rather than an emergent property.
Result types (`WriteResult`, `ReadResult`, `DeleteResult`) therefore carry `cost_usd`,
`latency_ms`, and `write_level` from day one so audit logging needs no later signature changes.

## Consequences

- Phase 0 ships contracts only; all tier behavior is implemented against frozen signatures.
- Backends never see policy; they store and retrieve. Policy decisions (thresholds, weights,
  decay) live in `MemoryPolicy` and the orchestration layer.
- `ForgetScope.ENTITY` is reserved now so GDPR erasure does not require a breaking change.
- The `read(tier, query, top_k)` signature assumes relevance-ranked retrieval; backends
  without semantic search may implement it as filtered substring/recency match.
