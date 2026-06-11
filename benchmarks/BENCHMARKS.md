# Mnemo benchmarks & comparison

Reproducible evaluation against **LongMemEval** and published scores from other memory systems.

## Metric: R@5 (session recall)

For each of 500 questions:

1. Ingest all `haystack_sessions` with `session_id` metadata.
2. `recall(question, top_k=5)`.
3. **Hit** if any retrieved item's `session_id` is in `answer_session_ids`.

```
R@5 = hits / questions
```

## Quick start (Windows PowerShell)

```powershell
cd orbit-mnemo
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,local-embeddings]"

# Download dataset (~140MB)
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest `
  -Uri "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" `
  -OutFile "data\longmemeval_s_cleaned.json"

# Smoke (2 turns)
python benchmarks\longmemeval_bench.py

# Mnemo dev subset (25 questions) — raw episodic path
python benchmarks\longmemeval_runner.py data\longmemeval_s_cleaned.json --limit 25 --mode raw --embedder hash

# Semantic embedder (slower, closer to MemPalace raw)
python benchmarks\longmemeval_runner.py data\longmemeval_s_cleaned.json --limit 25 --mode raw --embedder minilm
```

Full 500-question run (hours):

```powershell
python benchmarks\longmemeval_runner.py data\longmemeval_s_cleaned.json --mode raw --embedder minilm
```

## Mnemo modes

| Mode | Ingest | Compare to |
|------|--------|------------|
| `raw` | Episodic verbatim + embed only | MemPalace raw (96.6% R@5) |
| `full` | `Mnemo.remember()` all tiers + L0/L1/L2 | Mnemo production path |

## Published comparison (LongMemEval)

**Read this before quoting numbers.** Two different metrics appear in the wild:

| Metric | What it measures | Who uses it |
|--------|------------------|-------------|
| **R@5 (session recall)** | Is a gold `answer_session_id` in top-5 retrieved chunks? | MemPalace raw, **Mnemo runner** |
| **End-to-end QA accuracy** | Is the generated answer judged correct? | Mem0 (94.4%), Mastra (94.87%), Zep (~80%) |

These are **not directly comparable**. A system can have 100% R@5 and low QA accuracy.

### Retrieval recall (R@5) — apples-to-apples with Mnemo `--mode raw`

| System | R@5 | LLM at index | LLM at query | Source |
|--------|-----|--------------|--------------|--------|
| MemPalace hybrid v4 + rerank | 100% (500q) | No | Optional Haiku | [MemPalace](https://github.com/MemPalace/mempalace/blob/main/benchmarks/BENCHMARKS.md) |
| MemPalace hybrid v4 held-out | 98.4% (450q) | No | No | MemPalace |
| **MemPalace raw (Chroma)** | **96.6%** (500q) | **No** | **No** | MemPalace |
| agentmemory BM25+vector | ~95.2% | No | No | community |
| Stella dense retriever | ~85% | No | No | academic baseline |
| BM25 sparse | ~70% | No | No | academic baseline |

### End-to-end QA (different metric — do not mix with R@5)

| System | Score | LLM | Notes | Source |
|--------|-------|-----|-------|--------|
| Mem0 (2026 algorithm) | 94.4% | Yes | Token-efficient extraction | [mem0.ai/blog](https://mem0.ai/blog/state-of-ai-agent-memory-2026) |
| Mastra observational memory | 94.87% | GPT-5-mini | QA accuracy | Mastra research |
| Oracle GPT-4o (gold sessions only) | ~92% | Yes | Upper bound, not a product | LongMemEval paper |
| EverMemOS | 82.0% overall | Yes | Per-category breakdown published | EverMemOS eval |
| Zep (gpt-4o) | ~80% | Yes | +18.5% vs baseline in paper | arXiv:2501.13956 |
| Mem0 (older, 2025) | ~49–69% | Yes | Pre-2026 algorithm | Zep / EverMemOS tables |

### Why MemPalace raw beats Mem0 on retrieval

Mem0 **extracts** facts with an LLM and discards verbatim context. MemPalace/Mnemo **raw** keep every turn and search embeddings — nothing is lost before retrieval. On ConvoMem, MemPalace reports ~93% vs Mem0 RAG ~30–45% (different benchmark; same pattern).

## Mnemo measured results

Per-question audit trail: `benchmarks/results/*.jsonl`

| Run | n | Mode | Embedder | R@5 | avg ingest | avg recall | File |
|-----|---|------|----------|-----|------------|------------|------|
| dev | 5 | raw | MiniLM-L6-v2 | **100%** | 34.2 s | 72 ms | `mnemo_raw_minilm_n5.jsonl` |
| dev | 10 | raw | HashEmbedder | 10% | 72 ms | 26 ms | `mnemo_raw_hash_n10.jsonl` |
| dev | 25 | raw | MiniLM-L6-v2 | **100%** | 11.5 s | 47 ms | `mnemo_raw_minilm_n25.jsonl` |
| **full** | **500** | **raw** | **MiniLM-L6-v2** | **94.4%** | **12.5 s** | **49 ms** | `mnemo_raw_minilm_n500.jsonl` |

HashEmbedder is for **CI smoke only** — not semantic search. Use `--embedder minilm` for MemPalace-comparable runs.

**Fair MemPalace comparison:** complete 500q with same metric, document embedder (`all-MiniLM-L6-v2` vs Chroma default), use 50/450 hold-out split before claiming parity.

## Honest release checklist

- [x] Full 500q raw MiniLM run committed (`mnemo_raw_minilm_n500.jsonl`)
- [ ] Report dev (50q) vs hold-out (450q) split — do not tune on full 500
- [ ] Commit result JSONL with per-question audit trail
- [ ] Document embedder model (`all-MiniLM-L6-v2` vs MemPalace Chroma default)
- [ ] Report ingest + recall latency and write cost for `full` mode
- [ ] Do not claim SOTA without same dataset split and metric

## MemPalace repro (external baseline)

```bash
git clone https://github.com/MemPalace/mempalace.git
pip install -e mempalace
python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json
```
