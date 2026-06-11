# Mnemo

**Status: Phase 0 (foundation + contracts only). No runnable memory engine yet.**

Mnemo is a production-grade, backend-agnostic, cost-aware memory library for LLM agents.
It is framework-agnostic by design: the core has **no** LangChain, LangGraph, or other
agent-framework imports.

## The four tiers

| Tier | Purpose | Key behavior |
|------|---------|--------------|
| `working` | Bounded in-context memory | Importance-weighted FIFO eviction via min-heap on `(importance, arrival_time, key)` |
| `episodic` | Verbatim, timestamped events | Bi-temporal metadata; never summarized at write time |
| `semantic` | Extracted facts | Validity windows + confidence scoring |
| `procedural` | Behavioral patterns | Promoted only after N confirmed successes |

## Cost-aware write path (later phases)

Writes escalate through a cost ladder and stop at the cheapest sufficient level:

1. **Level 0** - regex extraction (free)
2. **Level 1** - template embedding match (cheap)
3. **Level 2** - LLM extraction, gated by KL-divergence threshold (expensive)

## Public API (end state, not yet implemented)

```python
await mnemo.remember(turn, policy)   # -> WriteResult
await mnemo.recall(query, policy)    # -> list[MemoryItem]
await mnemo.forget(entity, scope)    # -> None
```

## Install

```bash
pip install -e ".[dev]"
pytest -v
```

## What exists today (Phase 0)

- Core models: `MemoryTier`, `ForgetScope`, `MemoryItem`
- Result types with cost/audit fields: `WriteResult`, `ReadResult`, `DeleteResult`
- `MemoryPolicy` stub and `load_policy` skeleton
- `MemoryBackend` ABC (`write`, `read`, `delete`, `list`)
- Contract tests

See `docs/ADR-001-four-tier-architecture.md` for the architecture rationale and
`notes/BUILD_LOG.md` for the phase-by-phase build log.
