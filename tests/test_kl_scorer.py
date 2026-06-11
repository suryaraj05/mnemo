"""Phase 10: KL surprise scorer."""

from __future__ import annotations

from mnemo.scoring.kl_scorer import kl_surprise


def test_paraphrase_low_kl() -> None:
    vec = [1.0, 0.0, 0.0]
    assert kl_surprise(vec, [vec]) == 0.0


def test_novel_fact_high_kl() -> None:
    utterance = [0.0, 0.0, 1.0]
    beliefs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert kl_surprise(utterance, beliefs) > 0.15


def test_empty_beliefs_is_novel() -> None:
    assert kl_surprise([1.0, 0.0], []) == 1.0
