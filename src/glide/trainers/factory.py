"""Task -> trainer dispatch."""

from ..config.schema import GlideConfig, Task

__all__ = ["build_trainer"]


def build_trainer(config: GlideConfig):
    """Return a ready-to-train trainer for ``config.task``.

    * ``sft`` -> :func:`glide.trainers.sft.build_sft_trainer`
    * ``grpo`` / ``gspo`` -> :func:`glide.trainers.rl.build_rl_trainer`
    """
    if config.task is Task.SFT:
        from .sft import build_sft_trainer

        return build_sft_trainer(config)
    if config.task in (Task.GRPO, Task.GSPO):
        from .rl import build_rl_trainer

        return build_rl_trainer(config)
    raise ValueError(f"Unknown task: {config.task}")
