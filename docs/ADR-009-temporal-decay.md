# ADR-009: Temporal decay at retrieval

## Status

Accepted (Phase 9)

## Context

Memories should fade at recall time without deleting stored rows. Two candidate curves
from the foundations track: exponential (fast early drop) vs power-law (heavy tail).

## Decision

1. **Formulas** (age `t` in days):
   - Exponential: `w_exp(t) = e^(-λt)` with `λ = ln(2) / half_life_days`
   - Power-law: `w_pl(t) = (1 + t/τ)^(-α)`
2. **`DecayMode`** per tier on `MemoryPolicy`:
   - `episodic_decay_mode` — default `none` (backward compatible)
   - `semantic_decay_mode` — default `none`
3. **Episodic `recall_semantic`**: `score = cosine × decay(event_time age)`.
4. **Episodic `recall_recent`**: rank by decay weight when mode ≠ `none`.
5. **Semantic `recall_facts`**: rank by `decay(valid_from) × confidence`.
6. **Working / procedural**: no decay (eviction / promotion handle freshness).
7. **`scripts/decay_benchmark.py`**: table + optional matplotlib plot for tuning.

## Consequences

- With defaults `half_life=30`, `τ=7`, `α=1`, exponential weight exceeds power-law at `t=10`.
- Production agents should opt in via `episodic_decay_mode=exponential` explicitly.
- `get_timeline` and `get_history` remain undecayed audit views.
