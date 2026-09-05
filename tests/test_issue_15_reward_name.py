"""Regression test for #15: reward functions are named by their spec name."""

from glide.config.schema import GlideConfig, RewardSpec


def test_reward_functions_named_by_spec():
    import glide.plugins  # noqa: F401  registers built-ins
    from glide.trainers.common import build_reward_funcs

    cfg = GlideConfig()
    cfg.rl.rewards = [RewardSpec(name="format"), RewardSpec(name="cer")]
    funcs, weights = build_reward_funcs(cfg)
    # Each closure is renamed to its spec name (was the colliding "_reward").
    assert [f.__name__ for f in funcs] == ["format", "cer"]
