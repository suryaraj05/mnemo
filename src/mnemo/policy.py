"""Memory policy configuration (Phase 0 stub).

:class:`MemoryPolicy` will become the single knob surface controlling write
escalation, recall blending, and eviction. Phase 0 ships an intentionally
empty model so the ``remember/recall/forget`` signatures can stabilize before
any thresholds exist.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MemoryPolicy(BaseModel):
    """Tunable policy for Mnemo's write, recall, and eviction behavior.

    Phase 0: no fields yet. Planned fields (Phase 1+), kept here as the
    canonical roadmap:

    .. code-block:: python

        # max_working_size: int          # WORKING-tier bound before eviction
        # kl_threshold: float            # KL gate before escalating to L2 (LLM)
        # l1_cosine_threshold: float     # cosine cutoff for L1 template matches
        # tier_weights: dict[MemoryTier, float]  # recall blending weights
        # decay_half_life_days: float    # importance decay for eviction scoring
        # procedural_min_successes: int  # N successes before pattern promotion
    """

    model_config = ConfigDict(extra="forbid")


def load_policy(path: str) -> MemoryPolicy:
    """Load a :class:`MemoryPolicy` from a YAML file.

    Phase 0: not implemented. YAML loading lands together with the policy
    engine in a later phase to avoid taking a YAML dependency prematurely.

    Args:
        path: Filesystem path to a YAML policy document.

    Raises:
        NotImplementedError: Always, in Phase 0.
    """
    raise NotImplementedError(
        f"YAML policy loading is scheduled for a later phase (requested: {path!r}). "
        "Construct MemoryPolicy() directly for now."
    )
