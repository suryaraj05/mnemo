#!/usr/bin/env python3
"""LongMemEval R@k benchmark for Mnemo (session-level recall)."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mnemo import ExactVectorBackend, HashEmbedder, InMemoryBackend, MemoryPolicy, Mnemo
from mnemo.backends.base import MemoryBackend
from mnemo.embeddings.base import Embedder
from mnemo.tiers.episodic import EpisodicMemory


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    hit: bool
    recall_ms: float
    ingest_ms: float
    gold_session_ids: list[str]
    retrieved_session_ids: list[str]


@dataclass
class BenchmarkSummary:
    system: str
    mode: str
    dataset: str
    questions: int
    top_k: int
    recall_at_k: float
    avg_recall_ms: float
    avg_ingest_ms: float
    timestamp: str


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("dataset must be a JSON list")
    return data


def _session_ids_from_items(items: list[Any]) -> list[str]:
    found: list[str] = []
    for item in items:
        sid = item.metadata.get("session_id")
        if sid is not None and sid not in found:
            found.append(str(sid))
    return found


def ingest_haystack_raw(
    episodic: EpisodicMemory,
    session_ids: list[str],
    sessions: list[list[dict[str, Any]]],
    embedder: Embedder,
) -> float:
    """Verbatim episodic ingest (MemPalace-comparable raw path)."""
    start = time.perf_counter()
    for session_id, turns in zip(session_ids, sessions):
        for turn in turns:
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
            if not content.strip():
                continue
            episodic.record(
                content,
                source=role,
                metadata={"session_id": session_id},
                embedder=embedder,
            )
    return (time.perf_counter() - start) * 1000.0


def ingest_haystack_full(
    mnemo: Mnemo,
    session_ids: list[str],
    sessions: list[list[dict[str, Any]]],
) -> float:
    """Full Mnemo remember() path (tiers + write ladder)."""
    start = time.perf_counter()
    for session_id, turns in zip(session_ids, sessions):
        for turn in turns:
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
            if not content.strip():
                continue
            mnemo.remember(content, source=role, metadata={"session_id": session_id})
    return (time.perf_counter() - start) * 1000.0


def evaluate_question(
    *,
    backend: MemoryBackend,
    embedder: Embedder,
    policy: MemoryPolicy,
    entry: dict[str, Any],
    top_k: int,
    mode: str,
) -> QuestionResult:
    session_ids: list[str] = list(entry["haystack_session_ids"])
    sessions: list[list[dict[str, Any]]] = list(entry["haystack_sessions"])
    gold_ids = {str(value) for value in entry["answer_session_ids"]}
    question = str(entry["question"])
    qid = str(entry.get("question_id", entry.get("id", question[:32])))
    qtype = str(entry.get("question_type", entry.get("type", "unknown")))

    if mode == "raw":
        episodic = EpisodicMemory(backend)
        ingest_ms = ingest_haystack_raw(episodic, session_ids, sessions, embedder)
        mnemo = Mnemo(backend, policy, embedder=embedder)
    else:
        mnemo = Mnemo(backend, policy, embedder=embedder)
        ingest_ms = ingest_haystack_full(mnemo, session_ids, sessions)

    start = time.perf_counter()
    result = mnemo.recall(question, top_k=top_k)
    recall_ms = (time.perf_counter() - start) * 1000.0

    retrieved = _session_ids_from_items(result.items)
    hit = any(session_id in gold_ids for session_id in retrieved)

    return QuestionResult(
        question_id=qid,
        question_type=qtype,
        hit=hit,
        recall_ms=recall_ms,
        ingest_ms=ingest_ms,
        gold_session_ids=sorted(gold_ids),
        retrieved_session_ids=retrieved,
    )


def run_benchmark(
    dataset_path: Path,
    *,
    top_k: int = 5,
    limit: int | None = None,
    seed: int = 42,
    mode: str = "raw",
    embedder_name: str = "hash",
    output_jsonl: Path | None = None,
) -> BenchmarkSummary:
    entries = _load_dataset(dataset_path)
    if limit is not None:
        random.Random(seed).shuffle(entries)
        entries = entries[:limit]

    if embedder_name == "hash":
        embedder: Embedder = HashEmbedder(384)
        system = f"mnemo_{mode}_hash"
    elif embedder_name == "minilm":
        from mnemo.embeddings.local import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        system = f"mnemo_{mode}_minilm"
    else:
        raise ValueError(f"unknown embedder: {embedder_name}")

    policy = MemoryPolicy(
        episodic_decay_mode="none",
        semantic_decay_mode="none",
    )
    backend: MemoryBackend = ExactVectorBackend()

    results: list[QuestionResult] = []
    for entry in entries:
        backend = ExactVectorBackend()
        results.append(
            evaluate_question(
                backend=backend,
                embedder=embedder,
                policy=policy,
                entry=entry,
                top_k=top_k,
                mode=mode,
            )
        )

    hits = sum(1 for row in results if row.hit)
    summary = BenchmarkSummary(
        system=system,
        mode=mode,
        dataset=str(dataset_path),
        questions=len(results),
        top_k=top_k,
        recall_at_k=hits / len(results) if results else 0.0,
        avg_recall_ms=sum(row.recall_ms for row in results) / len(results),
        avg_ingest_ms=sum(row.ingest_ms for row in results) / len(results),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(summary)) + "\n")
            for row in results:
                handle.write(json.dumps(asdict(row)) + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemo LongMemEval R@k benchmark")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/longmemeval_s_cleaned.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Subset size (e.g. 25 dev)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("raw", "full"), default="raw")
    parser.add_argument("--embedder", choices=("hash", "minilm"), default="hash")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/mnemo_longmemeval.jsonl"),
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            "Download with:\n"
            "  Invoke-WebRequest -Uri "
            "'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_s_cleaned.json' "
            "-OutFile data/longmemeval_s_cleaned.json"
        )

    summary = run_benchmark(
        args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        seed=args.seed,
        mode=args.mode,
        embedder_name=args.embedder,
        output_jsonl=args.output,
    )
    print("Mnemo LongMemEval benchmark")
    print(f"  system: {summary.system}")
    print(f"  questions: {summary.questions}")
    print(f"  R@{summary.top_k}: {summary.recall_at_k * 100:.1f}%")
    print(f"  avg_recall_ms: {summary.avg_recall_ms:.2f}")
    print(f"  avg_ingest_ms: {summary.avg_ingest_ms:.2f}")
    print(f"  results: {args.output}")


if __name__ == "__main__":
    main()
