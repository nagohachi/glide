"""Experiment logging helpers (wandb / tensorboard) and config snapshotting.

Most logging is delegated to 🤗 ``TrainingArguments.report_to`` (wandb /
tensorboard). This module wires the structured ``logging`` config into the
environment that those integrations read, and snapshots the fully-resolved config
into the versioned output directory for reproducibility.
"""

import logging
import os
import sys
from pathlib import Path

import yaml

from ..config.schema import GlideConfig

__all__ = ["get_logger", "init_logging", "setup_logging", "snapshot_config"]

#: Root logger name; every module logs under ``glide.<area>`` so the emitted
#: prefix says where a line came from (``[glide.cli]``, ``[glide.eval]``, ...).
ROOT_LOGGER = "glide"


def get_logger(area: str | None = None) -> logging.Logger:
    """Return the ``glide`` logger, or the ``glide.<area>`` child logger.

    Args:
        area: Short area name (``cli``, ``eval``, ``models``, ...). ``None``
            returns the root ``glide`` logger.
    """
    return logging.getLogger(ROOT_LOGGER if area is None else f"{ROOT_LOGGER}.{area}")


def init_logging(level: int = logging.INFO) -> None:
    """Attach a stderr handler to the ``glide`` logger. Idempotent.

    Formats as ``[glide.<area>] message``, matching the ``[glide]`` prefix the
    CLI printed before. Only the ``glide`` logger is touched, so importing glide
    as a library never reconfigures the root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


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
