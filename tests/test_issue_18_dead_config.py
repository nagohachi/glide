"""Regression test for #18: dead config fields removed / wired."""

import dataclasses

from glide.config.schema import GlideConfig, RLConfig, Task


def test_ppo_not_advertised_and_reward_model_removed():
    # `task: ppo` was advertised but raises; the Task enum must not offer it.
    assert not any(t.value == "ppo" for t in Task)
    # `rl.reward_model` had no consumers -> removed from the schema.
    assert "reward_model" not in {f.name for f in dataclasses.fields(RLConfig)}


def test_data_num_workers_is_wired():
    from trl import SFTConfig

    from glide.config.loader import build_training_args

    cfg = GlideConfig()
    cfg.data.num_workers = 7
    cfg.training = {"output_dir": "x", "no_version": True, "bf16": False}
    args = build_training_args(cfg, SFTConfig)
    assert args.dataloader_num_workers == 7  # was never read before
