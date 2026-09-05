"""Launch topology and experiment logging."""

from dataclasses import dataclass, field

__all__ = ["DistributedConfig", "LoggingConfig"]


@dataclass
class DistributedConfig:
    """Multi-GPU / multi-node launch settings (driven from YAML, not env vars).

    ``glide <task>`` self-launches under ``torch.distributed.run`` when
    ``nproc_per_node`` resolves to > 1. ``nproc_per_node: null`` (default) means
    *auto* -- use all visible GPUs (``torch.cuda.device_count()``).
    """

    #: GPUs (processes) per node. ``None`` = auto-detect = number of visible GPUs.
    nproc_per_node: int | None = None
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str | None = None
    master_port: int | None = None


@dataclass
class LoggingConfig:
    """Experiment logging (wandb / tensorboard)."""

    #: Subset of ``["wandb", "tensorboard"]`` (or ``["none"]``).
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    project: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
