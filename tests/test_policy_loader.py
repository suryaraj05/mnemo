"""Phase 13: YAML policy loading."""

from __future__ import annotations

from importlib import resources

from mnemo.policy import load_policy


def test_load_voice_agent_example() -> None:
    path = resources.files("mnemo.policies.examples").joinpath("voice_agent.yaml")
    policy = load_policy(path)
    assert policy.max_working_size == 8
    assert policy.kl_threshold == 0.2
