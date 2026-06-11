# Metadata conventions (documentation only — not enforced)

`MemoryItem.metadata` is a free-form `dict[str, Any]`. These are the **reserved key
conventions** tier controllers will read and write in later phases. Backends MUST NOT
validate or depend on them; enforcement, if any, arrives with the tier controllers.

| Key | Tier | Meaning |
|------------|---------------------|--------------------------------------------------------|
| `importance` | WORKING | `float`, eviction score (higher = keep longer) |
| `t` | WORKING | `int`, monotonic arrival counter (heap tiebreaker) |
| `event_time` | EPISODIC | ISO 8601 datetime — when it happened in the world |
| `txn_from` | EPISODIC / SEMANTIC | ISO 8601 datetime — when Mnemo stored the record |
| `txn_to` | EPISODIC / SEMANTIC | ISO 8601 datetime or `null`; `null` = currently active |
| `source` | all tiers | `"user"` \| `"agent"` \| extractor id (e.g. `"l0:regex"`) |
| `scope` | EPISODIC | project/session namespace string |

Notes:

- `event_time` vs `txn_from`/`txn_to` implements bi-temporal storage: corrections close
  the old record (`txn_to` set) and insert a new one, never rewriting history.
- `t` exists because wall-clock timestamps can collide; the WORKING-tier eviction heap
  (Phase 2) orders on `(importance, t, key)`.
- Filter semantics for these keys follow ADR-002 `list()` rules: exact equality via
  `metadata.get(k) == v`.
