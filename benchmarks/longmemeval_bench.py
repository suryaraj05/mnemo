#!/usr/bin/env python3
"""LongMemEval-style smoke benchmark hook (Phase 19)."""

from __future__ import annotations

import argparse
import time

from mnemo import InMemoryBackend, Mnemo, MemoryPolicy


def run_smoke(top_k: int = 5) -> dict[str, float]:
    """Seed memories, recall, and report latency."""
    mnemo = Mnemo(InMemoryBackend(), MemoryPolicy())
    mnemo.remember("My name is Alex and I work on Mnemo.")
    mnemo.remember("I prefer Python for agent tooling.")

    start = time.perf_counter()
    result = mnemo.recall("What is my name?", top_k=top_k)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "hits": float(len(result.items)),
        "latency_ms": latency_ms,
        "tiers_searched": float(len(result.tiers_searched)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mnemo recall smoke benchmark")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    metrics = run_smoke(args.top_k)
    print("LongMemEval smoke benchmark")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
