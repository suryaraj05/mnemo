# Memory systems comparison (LongMemEval R@5)

**Purpose:** Open-source release positioning for Mnemo.  
**Metric:** Session-level Recall@5 on [LongMemEval-S](https://arxiv.org/abs/2410.10813) (500 questions, ~115k tokens/history).

---

## Headline table (published + Mnemo measured)

| Rank | System | R@5 | LLM at index? | LLM at query? | Local-first | Cost model |
|------|--------|-----|---------------|---------------|-------------|------------|
| 1 | MemPalace hybrid + rerank | ~99–100% | No | Optional | Yes | Free raw; ~$0.001/q rerank |
| 2 | Supermemory ASMR | ~99% | Yes | Yes | No | Paid API |
| 3 | MemPalace hybrid v4 | 98.4% | No | No | Yes | Free |
| 4 | **MemPalace raw** | **96.6%** | **No** | **No** | **Yes** | **Free** |
| 5 | Mastra | 94.87% | Yes | Yes | No | API |
| 6 | Hindsight | 91.4% | Yes | Yes | No | API |
| 7 | Mem0 (2026 QA) | 94.4% *QA* | Yes | Yes | Partial | API / SaaS |
| 8 | Zep (gpt-4o QA) | ~80% *QA* | Yes | Yes | Partial | $25+/mo |
| — | **Mnemo raw (MiniLM, n=5)** | **100%** | **No** | **No** | **Yes** | **Free** |
| — | **Mnemo raw (MiniLM, n=25 dev)** | **100%** | No | No | Yes | Free |
| — | Mnemo raw (HashEmbedder, n=10) | 10% | No | No | Yes | CI only — not semantic |
| — | Mnemo raw (MiniLM, n=500) | *pending* | No | No | Yes | ~25h CPU est. |

Sources: [MemPalace BENCHMARKS.md](https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md), LongMemEval paper.  
Mnemo rows: `benchmarks/results/*.jsonl` on this repo.

---

## What each system optimizes

| System | Architecture | Strength | Weakness |
|--------|--------------|----------|----------|
| **Mnemo** | 4 tiers + L0/L1/KL/L2 write ladder + hybrid recall | Cost-aware writes, bi-temporal semantic facts, backend-agnostic | Full 500q score pending; younger project |
| **MemPalace** | Verbatim Chroma + SQLite KG + hybrid boosts | Highest published R@5 without API | Monolithic storage; extraction N/A by design |
| **Mem0** | LLM extraction + multi-signal retrieval | Strong end-to-end QA (94.4%); low token budget | Different metric than R@5; loses verbatim |
| **Zep** | Temporal knowledge graph | Enterprise graph queries | QA ~80%; API cost |
| **Mastra** | LLM observations | Strong QA pipeline | Requires GPT-class model |
| **LongMemEval oracle** | Gold sessions only in context | Upper bound ~100% | Not a product |

---

## Mnemo differentiation (release narrative)

1. **Cost ladder** — L0 regex → L1 templates → KL-gated L2 LLM (`WriteResult.write_level`, `cost_usd`).
2. **Four tiers** — working / episodic / semantic / procedural in one library.
3. **Bi-temporal facts** — corrections close `txn_to` / `valid_to`, history preserved.
4. **Backend portability** — in-memory, SQLite, pgvector, Redis, composite.
5. **Framework-free** — no LangChain/LangGraph in core.

MemPalace wins on **mature hybrid retrieval tuning** today. Mnemo wins on **structured agent memory + cost control** — compare fairly on `--mode raw` first, then `--mode full` for write-path economics.

---

## Reproduce Mnemo numbers

```powershell
pip install -e ".[dev,local-embeddings]"
.\scripts\download_longmemeval.ps1
python benchmarks\longmemeval_runner.py data\longmemeval_s_cleaned.json --limit 25 --mode raw --embedder minilm
```

Full 500q (expect ~3 min/question ≈ 25 hours on CPU):

```powershell
python benchmarks\longmemeval_runner.py data\longmemeval_s_cleaned.json --mode raw --embedder minilm
```

---

## Release honesty rules

- Do **not** claim "beats MemPalace" until Mnemo completes **held-out 450q** with documented split.
- Report **embedder** (`all-MiniLM-L6-v2` vs Chroma default).
- Publish **per-question JSONL** under `benchmarks/results/`.
- Separate **raw** (retrieval-only) from **full** (production `remember()` path).

---

## Next engineering steps for competitive R@5

1. Hybrid recall tuning (keyword + temporal weights) — MemPalace v4 pattern.
2. Session-level chunking (index per session, not per turn).
3. Dev/hold-out split (`benchmarks/lme_split_50_450.json`).
4. Optional LLM rerank stage (document cost per query).
