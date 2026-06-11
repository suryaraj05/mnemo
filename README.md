# Mnemo

**Status: Phase 7 — Semantic tier + L0 regex extraction.**

Mnemo is a production-grade, backend-agnostic, cost-aware memory library for LLM agents.
It is framework-agnostic by design: the core has **no** LangChain, LangGraph, or other
agent-framework imports.

## The four tiers

| Tier | Purpose | Status |
|------|---------|--------|
| `working` | Bounded in-context memory | **Implemented** (`WorkingMemory`) |
| `episodic` | Verbatim, timestamped events | **Implemented** (`EpisodicMemory`) |
| `semantic` | Extracted facts (entity–predicate–value) | **Phase 7** |
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

## Semantic facts + L0 extraction (Phase 7)

```python
from mnemo import SemanticMemory, InMemoryBackend

sm = SemanticMemory(InMemoryBackend())
sm.ingest_l0("My name is Alex and I live in Pune.")
print(sm.get_fact("user", "name").value)  # Alex

# Correction preserves history (bi-temporal)
sm.store_fact("user", "name", "Alexander", source="user")
```

## Semantic episodic recall (Phase 6)

```python
from mnemo import ExactVectorBackend, HashEmbedder, EpisodicMemory

emb = HashEmbedder(dimension=384)
backend = ExactVectorBackend()
em = EpisodicMemory(backend)

em.record("I moved to Pune last month.", source="user", embedder=emb)
hits = em.recall_semantic("where did I live?", emb, top_k=3)
```

Production pgvector (requires Postgres + extension):

```bash
pip install -e ".[pgvector]"
export MNEMO_PGVECTOR_DSN=postgresql://localhost:5432/mnemo
```

## Embeddings (Phase 5)

```python
from mnemo import HashEmbedder, cosine_similarity

emb = HashEmbedder(dimension=384)  # CI-safe deterministic embedder
a, b = emb.embed("hello"), emb.embed("hello world")
score = cosine_similarity(a, b)
```

Local semantic model (optional):

```bash
pip install -e ".[dev,local-embeddings]"
```

```python
from mnemo.embeddings.local import SentenceTransformerEmbedder

emb = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
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
- `docs/ADR-005-embedding-layer.md`
- `docs/ADR-006-vector-retrieval.md`
- `docs/ADR-007-semantic-tier-l0.md`
- `docs/metadata-schema.md`
- `notes/BUILD_LOG.md`
