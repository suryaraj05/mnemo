"""Memory policy configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mnemo.models import DecayMode


def _default_tier_weights() -> dict[str, float]:
    return {
        "working": 0.7,
        "episodic": 0.85,
        "semantic": 1.0,
        "procedural": 0.75,
    }


class MemoryPolicy(BaseModel):
    """Tunable policy for Mnemo's write, recall, and eviction behavior."""

    model_config = ConfigDict(extra="forbid")

    max_working_size: int = Field(default=10, ge=1)
    l1_cosine_threshold: float = Field(default=0.85, ge=-1.0, le=1.0)
    l1_embed_cost_usd: float = Field(default=0.00001, ge=0.0)
    kl_threshold: float = Field(
        default=0.15,
        ge=0.0,
        description="Minimum KL surprise before L2 LLM extraction (Phase 10).",
    )
    l2_cost_usd: float = Field(default=0.01, ge=0.0)
    procedural_min_successes: int = Field(default=3, ge=1)
    episodic_decay_mode: DecayMode = Field(default=DecayMode.NONE)
    semantic_decay_mode: DecayMode = Field(default=DecayMode.NONE)
    decay_half_life_days: float = Field(default=30.0, gt=0.0)
    decay_tau_days: float = Field(default=7.0, gt=0.0)
    decay_alpha: float = Field(default=1.0, gt=0.0)
    tier_weights: dict[str, float] = Field(default_factory=_default_tier_weights)
    weight_semantic: float = Field(default=0.5, ge=0.0)
    weight_keyword: float = Field(default=0.3, ge=0.0)
    weight_temporal: float = Field(default=0.2, ge=0.0)

    @field_validator("tier_weights")
    @classmethod
    def _validate_tier_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for key in ("working", "episodic", "semantic", "procedural"):
            value.setdefault(key, _default_tier_weights()[key])
        return value


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "MNEMO_KL_THRESHOLD": ("kl_threshold", float),
        "MNEMO_MAX_WORKING_SIZE": ("max_working_size", int),
        "MNEMO_L1_COSINE_THRESHOLD": ("l1_cosine_threshold", float),
    }
    merged = dict(data)
    for env_key, (field_name, caster) in mapping.items():
        raw = os.environ.get(env_key)
        if raw is not None:
            merged[field_name] = caster(raw)
    return merged


def load_policy(path: str | Path) -> MemoryPolicy:
    """Load a :class:`MemoryPolicy` from a YAML file with optional env overrides."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError('load_policy requires PyYAML: pip install "mnemo[yaml]"') from exc

    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"policy file must contain a mapping, got {type(data)!r}")
    return MemoryPolicy.model_validate(_apply_env_overrides(data))
