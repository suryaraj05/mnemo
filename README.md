# Mnemo

**Status: Phase 4 — Working + Episodic tiers, InMemory + SQLite backends.**

Mnemo is a production-grade, backend-agnostic, cost-aware memory library for LLM agents.
It is framework-agnostic by design: the core has **no** LangChain, LangGraph, or other
agent-framework imports.

## The four tiers

| Tier | Purpose | Status |
|------|---------|--------|
| `working` | Bounded in-context memory | **Implemented** (`WorkingMemory`) |
| `episodic` | Verbatim, timestamped events | **Implemented** (`EpisodicMemory`) |
| `semantic` | Extracted facts | Planned |
| `procedural` | Behavioral patterns | Planned |

## Quick example

```python
from mnemo import InMemoryBackend, MemoryPolicy, WorkingMemory, EpisodicMemory

backend = InMemoryBackend()
wm = WorkingMemory.from_policy(backend, MemoryPolicy().max_working_size)
wm.add("turn_1", "User said hello", importance=0.9)

em = EpisodicMemory(backend)
em.record("I moved to Pune last month.", source="user", scope="my-agent")
recent = em.recall_recent(top_k=5)
```

### SQLite persistence

```python
from mnemo import SQLiteBackend, EpisodicMemory

em = EpisodicMemory(SQLiteBackend("mnemo.db"))
em.record("Survives restart", source="user")
```

## Install

```bash
pip install -e ".[dev]"
pytest -v
```

## Public API (end state)

```python
await mnemo.remember(turn, policy)   # -> WriteResult
await mnemo.recall(query, policy)    # -> list[MemoryItem]
await mnemo.forget(entity, scope)    # -> None
```

## Docs

- `docs/ADR-001-four-tier-architecture.md`
- `docs/ADR-002-inmemory-backend-semantics.md`
- `docs/ADR-003-working-memory-eviction.md`
- `docs/ADR-004-episodic-bi-temporal.md`
- `docs/metadata-schema.md`
- `notes/BUILD_LOG.md`
