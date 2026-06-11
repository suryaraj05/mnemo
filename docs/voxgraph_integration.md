# VoxGraph integration guide (Phase 18)

Mnemo core stays framework-free. Wire VoxGraph (or any agent) at the application layer.

## Hook map

| VoxGraph event | Mnemo call |
|----------------|------------|
| `ConversationTurn` received | `mnemo.remember(turn.text, source="user")` |
| Before LLM prompt | `mnemo.recall(query, top_k=policy.max_working_size)` |
| User erasure request | `mnemo.forget(ForgetScope.ENTITY, entity="user")` |
| Successful tool workflow | `mnemo.procedural.record_episode(task_type, steps, success=True, embedder)` |

## Example

```python
from mnemo import Mnemo, SQLiteBackend, load_policy, ForgetScope

mnemo = Mnemo(SQLiteBackend("voxgraph.db"), load_policy("voice_agent.yaml"))

def on_turn(text: str) -> None:
    mnemo.remember(text, source="user")

def build_context(user_message: str) -> str:
    hits = mnemo.recall(user_message, top_k=8)
    return "\n".join(f"- {item.value}" for item in hits.items)
```

## Checklist

- [ ] Episodic verbatim audit trail preserved
- [ ] Semantic facts extracted via L0/L1 before L2 cost
- [ ] Working tier bounded via policy
- [ ] No LangChain imports inside `mnemo` package
