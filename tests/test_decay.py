"""Phase 9 tests: temporal decay functions."""

from __future__ import annotations

import math

import pytest

from mnemo.decay import decay_weight, weight_exponential, weight_power_law
from mnemo.models import DecayMode


def test_exponential_half_life_is_half() -> None:
    assert weight_exponential(30.0, 30.0) == pytest.approx(0.5)


def test_exponential_at_zero_is_one() -> None:
    assert weight_exponential(0.0, 30.0) == pytest.approx(1.0)


def test_power_law_at_zero_is_one() -> None:
    assert weight_power_law(0.0, 7.0, 1.0) == pytest.approx(1.0)


def test_at_ten_days_exponential_larger_than_power_law() -> None:
    """Quiz default: half-life 30d, τ=7, α=1 → exp wins at t=10."""
    exp_w = weight_exponential(10.0, 30.0)
    pl_w = weight_power_law(10.0, 7.0, 1.0)
    assert exp_w > pl_w
    assert exp_w == pytest.approx(2 ** (-10 / 30))
    assert pl_w == pytest.approx((1 + 10 / 7) ** -1)


def test_decay_weight_none_is_identity() -> None:
    assert decay_weight(100.0, DecayMode.NONE) == 1.0


def test_lambda_matches_ln2_over_half_life() -> None:
    half_life = 20.0
    lam = math.log(2) / half_life
    assert weight_exponential(half_life, half_life) == pytest.approx(math.exp(-lam * half_life))
