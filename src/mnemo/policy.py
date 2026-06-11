"""Memory policy configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mnemo.models import DecayMode


class MemoryPolicy(BaseModel):
    """Tunable policy for Mnemo's write, recall, and eviction behavior.

    Planned fields (later phases):

    .. code-block:: python

        # kl_threshold: float
        # tier_weights: dict[str, float]
        # procedural_min_successes: int
    """

    model_config = ConfigDict(extra="forbid")

    max_working_size: int = Field(
        default=10,
        ge=1,
        description="Maximum items in WORKING tier before eviction (Phase 2).",
    )
    l1_cosine_threshold: float = Field(
        default=0.85,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity for L1 template match (Phase 8).",
    )
    l1_embed_cost_usd: float = Field(
        default=0.00001,
        ge=0.0,
        description="Marginal USD cost per query embed call on the L1 write path.",
    )
    episodic_decay_mode: DecayMode = Field(
        default=DecayMode.NONE,
        description="Decay curve for episodic recall (default off for compatibility).",
    )
    semantic_decay_mode: DecayMode = Field(
        default=DecayMode.NONE,
        description="Decay curve for semantic fact ranking at recall.",
    )
    decay_half_life_days: float = Field(
        default=30.0,
        gt=0.0,
        description="Half-life for exponential decay (days).",
    )
    decay_tau_days: float = Field(
        default=7.0,
        gt=0.0,
        description="Scale τ for power-law decay (days).",
    )
    decay_alpha: float = Field(
        default=1.0,
        gt=0.0,
        description="Exponent α for power-law decay.",
    )


def load_policy(path: str) -> MemoryPolicy:
    """Load a :class:`MemoryPolicy` from a YAML file (not yet implemented)."""
    raise NotImplementedError(
        f"YAML policy loading is scheduled for a later phase (requested: {path!r}). "
        "Construct MemoryPolicy() directly for now."
    )
