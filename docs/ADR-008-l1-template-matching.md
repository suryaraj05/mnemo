# ADR-008: L1 template embedding match

## Status

Accepted (Phase 8)

## Context

Level 0 regex misses paraphrases (`"I started at Stripe"` vs `"I work at Google"`).
Level 1 matches utterance shape via embeddings before paying for LLM extraction (L2).

## Decision

1. **`TemplateLibrary`**: JSON list of `FactTemplate` rows — `entity`, `predicate`,
   `utterances[]`, `value_pattern` (regex capture for the value slot).
2. **Bundled default** at `mnemo/extraction/data/l1_templates.json`.
3. **Matching**: embed input once; compare cosine to cached template utterance vectors;
   best score per slot must exceed `MemoryPolicy.l1_cosine_threshold` (default `0.85`).
4. **Value extraction**: winning slot's `value_pattern` applied to source text.
5. **Confidence**: `min(0.75, score)` — medium confidence, below L0 regex certainty.
6. **Cost**: `WriteResult.cost_usd = policy.l1_embed_cost_usd` per ingest (one query embed);
   template index embeds are amortized via `library.warm_cache()`.
7. **`SemanticMemory.ingest_l1`**: stores with `source="l1:template"`, `write_level=1`.

## Consequences

- Threshold tuning belongs on a dev set; default 0.85 ≈ 32° angle between vectors.
- Real semantic embedders (`SentenceTransformerEmbedder`) unlock production L1 quality;
  CI uses cluster stub embedders in tests.
- L2 KL gate (Phase 10+) runs only when L0 + L1 produce no facts.
