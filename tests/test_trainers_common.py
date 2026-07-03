"""Tests for trainer-common helpers."""

from glide.config.schema import GlideConfig, RLConfig, RewardSpec
from glide.trainers.common import build_reward_funcs


def test_build_reward_funcs_uses_spec_names_for_callable_names():
    config = GlideConfig(
        rl=RLConfig(
            rewards=[
                RewardSpec(name="exact_match", weight=0.7),
                RewardSpec(name="format", weight=1.3),
            ]
        )
    )

    funcs, weights = build_reward_funcs(config)

    assert [fn.__name__ for fn in funcs] == ["exact_match", "format"]
    assert weights == [0.7, 1.3]
