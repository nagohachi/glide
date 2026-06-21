"""Experiment logging helpers (wandb / tensorboard) and config snapshotting.

Most logging is delegated to 🤗 ``TrainingArguments.report_to`` (wandb /
tensorboard). This module wires the structured ``logging`` config into the
environment that those integrations read, and snapshots the fully-resolved config
into the versioned output directory for reproducibility.
"""

import os
from pathlib import Path

import yaml

from ..config.schema import GlideConfig

__all__ = ["setup_logging", "snapshot_config"]


def setup_logging(config: GlideConfig) -> None:
    """Configure wandb/tensorboard integrations from ``config.logging``."""
    log = config.logging
    if "wandb" in log.report_to:
        if log.project:
            os.environ.setdefault("WANDB_PROJECT", log.project)
        if log.tags:
            os.environ.setdefault("WANDB_TAGS", ",".join(log.tags))
        if log.run_name:
            os.environ.setdefault("WANDB_NAME", log.run_name)


def snapshot_config(config: GlideConfig, output_dir: str | os.PathLike) -> str:
    """Write the resolved config to ``output_dir/glide_config.yaml`` and return its path.

    This is the single source of truth for *what was actually run*, after YAML
    ``extends`` resolution and CLI overrides.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "glide_config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)
    return str(path)
