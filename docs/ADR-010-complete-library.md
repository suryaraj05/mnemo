# ADR-010: Complete library surface (Phases 10–19)

## Status

Accepted

## Decision

Phases 10–19 complete the v0.1.0 library:

| Phase | Deliverable |
|-------|-------------|
| 10–11 | `kl_surprise` gates L2 when beliefs explain utterance |
| 12 | `LLMExtractor` / `MockLLMExtractor`, write ladder in `pipeline.py` |
| 13 | `load_policy` YAML + example policies + env overrides |
| 14 | `ProceduralMemory` with `procedural_min_successes` |
| 15 | `Mnemo.recall` composite hybrid scoring |
| 16 | `Mnemo.forget` + `ForgetScope` across tiers |
| 17 | `RedisBackend` + `CompositeBackend` |
| 18 | `docs/voxgraph_integration.md` (app-layer hooks) |
| 19 | `benchmarks/longmemeval_bench.py` smoke script |

Public entry point: `Mnemo.remember / recall / forget`.

## Consequences

- Async API deferred; sync facade matches backend contract.
- Real LLM providers plug in via `LLMExtractor` subclass.
- Redis/pgvector integration tests remain env-gated in CI.
