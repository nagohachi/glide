"""Tests for the config system: extends, merge, CLI overrides, hydration, versioning."""

import datetime as dt
from pathlib import Path

import pytest

from glide.config import GlideConfig, Modality, Task, load_config, version_output_dir
from glide.trainers.common import build_reward_funcs, init_plugins
from glide.config.loader import (
    apply_overrides,
    build_training_args,
    deep_merge,
    dict_to_dataclass,
    parse_override_args,
)


def test_deep_merge_nested():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 3, "z": 4}, "c": 5}
    assert deep_merge(base, override) == {"a": 1, "b": {"x": 1, "y": 3, "z": 4}, "c": 5}


def test_parse_override_args_types_and_nesting():
    overrides = parse_override_args(
        ["--model.name", "Qwen/Q", "--training.learning_rate", "1e-5",
         "--peft.enabled", "--data.lazy=false", "--logging.report_to", '["wandb"]']
    )
    assert overrides["model"]["name"] == "Qwen/Q"
    assert overrides["training"]["learning_rate"] == 1e-5
    assert overrides["peft"]["enabled"] is True
    assert overrides["data"]["lazy"] is False
    assert overrides["logging"]["report_to"] == ["wandb"]


def test_extends_chain_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "task: sft\nmodel:\n  name: base-model\ntraining:\n  learning_rate: 2.0e-5\n"
    )
    (tmp_path / "run.yaml").write_text(
        "extends: base.yaml\nmodel:\n  name: run-model\ntraining:\n  num_train_epochs: 3\n"
    )
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.model.name == "run-model"  # child overrides parent
    assert cfg.training["learning_rate"] == 2.0e-5  # inherited
    assert cfg.training["num_train_epochs"] == 3


def test_cli_override_beats_yaml(tmp_path):
    (tmp_path / "run.yaml").write_text("model:\n  name: yaml-model\n")
    cfg = load_config(tmp_path / "run.yaml", ["--model.name", "cli-model"], task="grpo")
    assert cfg.model.name == "cli-model"
    assert cfg.task is Task.GRPO


def test_cyclic_extends_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown config key"):
        dict_to_dataclass(GlideConfig, {"model": {"nonexistent": 1}})


def test_enum_and_modality_hydration():
    cfg = dict_to_dataclass(GlideConfig, {"task": "gspo", "modality": "speech"})
    assert cfg.task is Task.GSPO
    assert cfg.modality is Modality.SPEECH


def test_reward_spec_list_hydration():
    cfg = dict_to_dataclass(
        GlideConfig,
        {"rl": {"rewards": [{"name": "cer", "weight": 1.0, "kwargs": {"reference_key": "ref"}}]}},
    )
    assert cfg.rl.rewards[0].name == "cer"
    assert cfg.rl.rewards[0].kwargs == {"reference_key": "ref"}


def test_version_output_dir_increments(tmp_path):
    now = dt.datetime(2026, 6, 21, 10, 30, 0)
    first = version_output_dir(tmp_path, now=now)
    assert Path(first).name == "v0-20260621-103000"
    Path(first).mkdir()
    (tmp_path / "v3-old").mkdir()
    second = version_output_dir(tmp_path, now=now)
    assert Path(second).name == "v4-20260621-103000"  # max(0,3)+1


def test_build_training_args_versions_and_filters(tmp_path):
    from trl import SFTConfig

    cfg = GlideConfig()
    cfg.training = {"output_dir": str(tmp_path / "out"), "learning_rate": 1e-5}
    args = build_training_args(cfg, SFTConfig, now=dt.datetime(2026, 1, 1, 0, 0, 0))
    assert Path(args.output_dir).name.startswith("v0-")
    assert args.learning_rate == 1e-5


def test_build_training_args_unknown_key_raises():
    from trl import SFTConfig

    cfg = GlideConfig()
    cfg.training = {"output_dir": "x", "not_a_real_arg": 1, "no_version": True}
    with pytest.raises(ValueError, match="Unknown `training` key"):
        build_training_args(cfg, SFTConfig)


def test_distributed_config_hydration_and_default():
    cfg = dict_to_dataclass(GlideConfig, {})
    assert cfg.distributed.nproc_per_node is None  # auto = all visible GPUs
    assert cfg.distributed.nnodes == 1
    cfg2 = dict_to_dataclass(
        GlideConfig, {"distributed": {"nproc_per_node": 4, "nnodes": 2, "node_rank": 1}}
    )
    assert cfg2.distributed.nproc_per_node == 4
    assert cfg2.distributed.nnodes == 2 and cfg2.distributed.node_rank == 1


def test_apply_overrides_roundtrip():
    merged = apply_overrides({"a": {"b": 1}}, ["--a.c", "2"])
    assert merged == {"a": {"b": 1, "c": 2}}


def test_data_corpus_resolves_root(tmp_path):
    (tmp_path / "run.yaml").write_text(
        "data_roots:\n  csj: /abs/csj\n  cv: /abs/cv\n"
        "data:\n  corpus: csj\n  train: a/train.jsonl\n  eval: a/dev.jsonl\n"
    )
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.data.train == "/abs/csj/a/train.jsonl"
    assert cfg.data.eval == "/abs/csj/a/dev.jsonl"


def test_data_corpus_unknown_raises(tmp_path):
    (tmp_path / "run.yaml").write_text(
        "data_roots:\n  csj: /abs/csj\n"
        "data:\n  corpus: missing\n  train: t.jsonl\n"
    )
    with pytest.raises(ValueError, match="not found in data_roots"):
        load_config(tmp_path / "run.yaml")


def test_data_root_explicit_still_works(tmp_path):
    (tmp_path / "run.yaml").write_text("data:\n  root: /abs/r\n  train: t.jsonl\n")
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.data.train == "/abs/r/t.jsonl"


def test_build_reward_funcs_uses_reward_spec_names():
    cfg = dict_to_dataclass(
        GlideConfig,
        {"rl": {"rewards": [{"name": "cer", "weight": 1.0}, {"name": "format", "weight": 1.0}]}},
    )

    init_plugins(cfg)

    funcs, _weights = build_reward_funcs(cfg)
    assert [fn.__name__ for fn in funcs] == ["cer", "format"]
