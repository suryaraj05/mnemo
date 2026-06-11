# Mnemo

**Status: v0.1.0 — Complete 4-tier library with cost-aware write path and public API.**

Mnemo is a production-grade, backend-agnostic, cost-aware memory library for LLM agents.
It is framework-agnostic by design: the core has **no** LangChain, LangGraph, or other
agent-framework imports.

## The four tiers

| Tier | Purpose | Status |
|------|---------|--------|
| `working` | Bounded in-context memory | **Done** |
| `episodic` | Verbatim, timestamped events | **Done** |
| `semantic` | Extracted facts (entity–predicate–value) | **Done** |
| `procedural` | Behavioral patterns after N successes | **Done** |

## Public API

```python
from mnemo import Mnemo, InMemoryBackend, MemoryPolicy, ForgetScope, load_policy

mnemo = Mnemo(InMemoryBackend(), load_policy("path/to/voice_agent.yaml"))

evt_key, pipeline = mnemo.remember("My name is Alex and I live in Pune.")
print(pipeline.write_level)  # 0 = regex, 1 = template, 2 = LLM

hits = mnemo.recall("where do I live?", top_k=5)
mnemo.forget(ForgetScope.ENTITY, entity="user")
```

Write path ladder: **L0 regex → L1 template embed → KL gate → L2 LLM**.

## Install

```bash
pip install -e ".[dev]"
pytest -v
```

Optional extras:

```bash
pip install -e ".[local-embeddings]"   # SentenceTransformer
pip install -e ".[pgvector]"           # Postgres ANN
pip install -e ".[redis]"              # Redis working tier
```

## Example policies

Bundled under `src/mnemo/policies/examples/`:

- `voice_agent.yaml`
- `research_agent.yaml`
- `support_agent.yaml`

## Benchmarks

```bash
python scripts/decay_benchmark.py
python benchmarks/longmemeval_bench.py
```

## Backends

| Backend | Use case |
|---------|----------|
| `InMemoryBackend` | Tests, notebooks |
| `SQLiteBackend` | Local persistent agents |
| `ExactVectorBackend` | CI semantic search |
| `PgvectorBackend` | Production ANN |
| `RedisBackend` | Low-latency working tier |
| `CompositeBackend` | e.g. Redis working + SQLite episodic |

## Docs

- ADR-001 … ADR-009 (architecture through decay)
- `docs/ADR-010-complete-library.md`
- `docs/voxgraph_integration.md`
- `docs/metadata-schema.md`
- `notes/BUILD_LOG.md`
